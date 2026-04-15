#include "cpu_gen.h"
#include "cpu_util.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// `CAST_BLOCK_GEMM_T` — experimental blocked matvec.
//
// Unset / "" / "0"  → legacy straight-line matvec (default).
// "4" / "8" / "16"  → emit matvec as blocks of T output rows with explicit
//                      basic-block boundaries and `store volatile` /
//                      `load volatile` through a stack-allocated scratch
//                      buffer that retires block accumulators between
//                      blocks (breaks the matvec's live-range pressure
//                      that currently peaks at ~4·K vector values at
//                      k ≥ 5).
//
// Only engages when `K > T` (so the blocking actually subdivides the
// matvec).  Orthogonal to the lo-heavy load tiling — both may apply.
// ---------------------------------------------------------------------------
static unsigned get_block_gemm_tile() {
  const char *s = std::getenv("CAST_BLOCK_GEMM_T");
  if (!s || s[0] == '\0')
    return 0;
  char *end = nullptr;
  const long v = std::strtol(s, &end, 10);
  if (!end || *end != '\0' || v <= 0)
    return 0;
  // Accept only powers of two in [2, 64]; anything else silently falls back
  // to legacy to avoid footguns.
  if (v < 2 || v > 64 || (v & (v - 1)) != 0)
    return 0;
  return static_cast<unsigned>(v);
}

// ---------------------------------------------------------------------------
// Basic types
// ---------------------------------------------------------------------------

struct LaunchArgs {
  llvm::Value *p_sv = nullptr;
  llvm::Value *ctr_begin = nullptr;
  llvm::Value *ctr_end = nullptr;
  llvm::Value *p_mat = nullptr;
};

struct MatrixView {
  const cast_cpu_complex64_t *data = nullptr;
  uint32_t edge_size = 0;

  double re(size_t idx) const { return data[idx].re; }
  double im(size_t idx) const { return data[idx].im; }
};

// One re/im vector pair per matrix element.
struct IRMatData {
  llvm::Value *re_vec = nullptr;
  llvm::Value *im_vec = nullptr;
};

// ---------------------------------------------------------------------------
// Bit layout
//
// Qubits are partitioned into three groups relative to the SIMD register size
// (simd_s = log2 of lanes):
//   simd_bits  — non-target positions that form the SIMD lanes
//   lo_bits    — target qubits whose positions fall within the SIMD range
//                (handled via shuffles inside one register load)
//   hi_bits    — target qubits whose positions are above the SIMD range
//                (each combination requires a separate aligned load)
//
// All positions are adjusted +1 for those at or above simd_s to account for
// the implicit zero bit the statevector layout inserts there.
// sep_bit is the first bit position above the SIMD register region.
// ---------------------------------------------------------------------------

struct BitLayout {
  std::vector<unsigned> simd_bits;
  std::vector<unsigned> lo_bits;
  std::vector<unsigned> hi_bits;
  unsigned sep_bit = 0;
};

static BitLayout compute_bit_layout(const uint32_t *qubits, size_t n_qubits, unsigned simd_s) {
  BitLayout layout;
  unsigned q = 0;
  size_t qi = 0;

  while (layout.simd_bits.size() < simd_s) {
    if (qi < n_qubits && qubits[qi] == q) {
      layout.lo_bits.push_back(q);
      ++qi;
    } else {
      layout.simd_bits.push_back(q);
    }
    ++q;
  }
  while (qi < n_qubits) {
    layout.hi_bits.push_back(qubits[qi++]);
  }

  for (auto *vec : {&layout.lo_bits, &layout.simd_bits, &layout.hi_bits}) {
    for (auto &bit : *vec) {
      if (bit >= simd_s)
        ++bit;
    }
  }

  layout.sep_bit = layout.simd_bits.empty() ? 0u : layout.simd_bits.back() + 1u;
  if (layout.sep_bit == simd_s)
    ++layout.sep_bit;

  return layout;
}

// ---------------------------------------------------------------------------
// Statevector base-pointer computation
//
// Each task_id indexes the "free" dimensions of the statevector (those not
// occupied by hi_bits).  The actual vec_ty* pointer is obtained by scattering
// task_id bits into the non-hi positions.
// ---------------------------------------------------------------------------

struct PtrSegment {
  uint64_t src_mask;
  unsigned dst_shift;
};

static std::vector<PtrSegment> compute_hi_ptr_segments(const std::vector<unsigned> &hi_bits,
                                                       unsigned sep_bit) {
  std::vector<PtrSegment> segs;
  unsigned src_bit = 0;
  unsigned prev_end = 0;

  for (const unsigned hb : hi_bits) {
    const unsigned hi_pos = hb - sep_bit;
    if (hi_pos > prev_end) {
      const unsigned width = hi_pos - prev_end;
      const uint64_t mask = ((uint64_t(1) << width) - 1) << src_bit;
      segs.push_back({mask, prev_end - src_bit});
      src_bit += width;
    }
    prev_end = hi_pos + 1;
  }

  segs.push_back({~((uint64_t(1) << src_bit) - 1), prev_end - src_bit});
  return segs;
}

// ---------------------------------------------------------------------------
// Parallel bit deposit (used for split-mask precomputation only)
// ---------------------------------------------------------------------------

static uint32_t pdep32(uint32_t src, uint32_t mask, unsigned nbits = 32) {
  uint32_t out = 0;
  unsigned src_bit = 0;
  for (unsigned dst_bit = 0; dst_bit < nbits; ++dst_bit) {
    if (!(mask & (uint32_t(1) << dst_bit)))
      continue;
    if (src & (uint32_t(1) << src_bit))
      out |= uint32_t(1) << dst_bit;
    ++src_bit;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Shuffle masks
// ---------------------------------------------------------------------------

struct ShuffleMasks {
  std::vector<int> re_split;
  std::vector<int> im_split;
  std::vector<std::vector<int>> merge;
  std::vector<int> reim_merge;
};

static ShuffleMasks compute_shuffle_masks(const BitLayout &layout, unsigned s, unsigned simd_s,
                                          unsigned vec_size) {
  const unsigned LK = 1u << layout.lo_bits.size();
  ShuffleMasks out;

  // --- Split masks: extract re/im lanes for each lo-qubit partition ---
  out.re_split.resize(LK * s);
  out.im_split.resize(LK * s);
  {
    uint32_t pdep_mask_s = 0;
    const unsigned pdep_nbits_s = layout.simd_bits.empty() ? 0u : layout.simd_bits.back() + 1u;
    for (auto bit : layout.simd_bits)
      pdep_mask_s |= uint32_t(1) << bit;

    uint32_t pdep_mask_l = 0;
    const unsigned pdep_nbits_l = layout.lo_bits.empty() ? 0u : layout.lo_bits.back() + 1u;
    for (auto bit : layout.lo_bits)
      pdep_mask_l |= uint32_t(1) << bit;

    for (unsigned li = 0; li < LK; ++li) {
      for (unsigned si = 0; si < s; ++si) {
        out.re_split[li * s + si] = static_cast<int>(pdep32(li, pdep_mask_l, pdep_nbits_l) |
                                                     pdep32(si, pdep_mask_s, pdep_nbits_s));
        out.im_split[li * s + si] = out.re_split[li * s + si] | (1 << simd_s);
      }
    }
  }

  // --- Merge masks: iterative merge-sort for reassembling lo-partitions ---
  {
    std::vector<int> current(out.re_split.begin(), out.re_split.begin() + s);
    for (size_t round = 0; round < layout.lo_bits.size(); ++round) {
      const int half = static_cast<int>(current.size());

      std::vector<int> rhs(half);
      for (int i = 0; i < half; ++i)
        rhs[i] = current[i] | (1 << layout.lo_bits[round]);

      out.merge.emplace_back(half * 2);
      auto &mask = out.merge.back();
      std::vector<int> merged(half * 2);
      int il = 0, ir = 0, im = 0;
      while (il < half || ir < half) {
        const bool take_left = (ir == half) || (il < half && current[il] < rhs[ir]);
        if (take_left) {
          mask[im] = il;
          merged[im++] = current[il++];
        } else {
          mask[im] = ir + half;
          merged[im++] = rhs[ir++];
        }
      }
      current = std::move(merged);
    }
  }

  // --- Re-im interleave mask: [re0, im0, re1, im1, ...] ---
  out.reim_merge.reserve(vec_size);
  for (unsigned pair_idx = 0; pair_idx < (vec_size >> simd_s >> 1); ++pair_idx) {
    for (unsigned i = 0; i < s; ++i)
      out.reim_merge.push_back(static_cast<int>(s * pair_idx + i));
    for (unsigned i = 0; i < s; ++i)
      out.reim_merge.push_back(static_cast<int>(s * pair_idx + i + (vec_size >> 1)));
  }

  return out;
}

// ---------------------------------------------------------------------------
// Matrix data: emit splatted constant vectors (ImmValue) or runtime
// load+splat (StackLoad) for each matrix element.
// ---------------------------------------------------------------------------

static std::vector<IRMatData> build_matrix_data(llvm::IRBuilder<> &builder,
                                                const cast_cpu_kernel_gen_spec_t &spec,
                                                const MatrixView &matrix, llvm::Value *p_mat_arg,
                                                unsigned simd_s) {
  const unsigned kk = matrix.edge_size * matrix.edge_size;
  auto *scalar_ty =
      (spec.precision == CAST_CPU_PRECISION_F32) ? builder.getFloatTy() : builder.getDoubleTy();
  const auto ec = llvm::ElementCount::getFixed(1u << simd_s);

  std::vector<IRMatData> out(kk);
  for (unsigned i = 0; i < kk; ++i) {
    if (spec.mode == CAST_CPU_MATRIX_LOAD_IMM_VALUE) {
      out[i].re_vec =
          llvm::ConstantVector::getSplat(ec, llvm::ConstantFP::get(scalar_ty, matrix.re(i)));
      out[i].im_vec =
          llvm::ConstantVector::getSplat(ec, llvm::ConstantFP::get(scalar_ty, matrix.im(i)));
    } else {
      auto *re_ptr = builder.CreateConstGEP1_32(scalar_ty, p_mat_arg, 2 * i, "re.mat.ptr");
      out[i].re_vec = builder.CreateVectorSplat(ec, builder.CreateLoad(scalar_ty, re_ptr, "re.mat"),
                                                "re.mat.vec");
      auto *im_ptr = builder.CreateConstGEP1_32(scalar_ty, p_mat_arg, 2 * i + 1, "im.mat.ptr");
      out[i].im_vec = builder.CreateVectorSplat(ec, builder.CreateLoad(scalar_ty, im_ptr, "im.mat"),
                                                "im.mat.vec");
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Kernel IR generation context
//
// Bundles the derived constants, LLVM types, and precomputed data needed by
// the IR emission helpers below.  Constructed once in the public entry point
// and passed by reference to each phase.
// ---------------------------------------------------------------------------

struct KernelCodegen {
  llvm::IRBuilder<> &B;

  // Gate geometry
  unsigned k;        // number of gate qubits
  unsigned K;        // matrix dimension = 2^k
  unsigned lk;       // lo-qubit count
  unsigned LK;       // lo-qubit combinations = 2^lk
  unsigned hk;       // hi-qubit count
  unsigned HK;       // hi-qubit combinations = 2^hk
  unsigned s;        // SIMD lanes = 2^simd_s
  unsigned simd_s;   // log2(SIMD lanes)
  unsigned vec_size; // interleaved vector width = 2^sep_bit

  unsigned simd_width_bytes;

  const BitLayout &layout;
  const ShuffleMasks &smasks;
  const std::vector<IRMatData> &mat_data;

  llvm::Type *scalar_ty;
  llvm::Type *vec_ty;

  // Blocked-matvec experiment: 0 = disabled (legacy straight-line matvec);
  // T > 0 = block output rows in groups of T.  Only applied when LK > T.
  unsigned block_gemm_t;
  llvm::Value *mv_re_scratch = nullptr; // set iff block_gemm_t > 0 && LK > block_gemm_t
  llvm::Value *mv_im_scratch = nullptr;
};

// ---------------------------------------------------------------------------
// Phase 1: Load amplitudes
//
// For each hi-qubit combination, do one aligned vector load and shuffle-split
// into separate re/im vectors for each lo-qubit partition.
// ---------------------------------------------------------------------------

struct LoadedAmplitudes {
  std::vector<llvm::Value *> re;   // K elements: re[hi * LK + li]
  std::vector<llvm::Value *> im;   // K elements
  std::vector<llvm::Value *> ptrs; // HK pointers (for stores later)
};

static LoadedAmplitudes emit_load_amplitudes(KernelCodegen &cg, llvm::Value *ptr_sv_begin) {
  LoadedAmplitudes amps;
  amps.re.resize(cg.K);
  amps.im.resize(cg.K);
  amps.ptrs.resize(cg.HK);

  for (unsigned hi = 0; hi < cg.HK; ++hi) {
    uint64_t idx_shift = 0;
    for (unsigned hbit = 0; hbit < cg.hk; ++hbit) {
      if (hi & (1u << hbit))
        idx_shift += uint64_t(1) << cg.layout.hi_bits[hbit];
    }
    idx_shift >>= cg.layout.sep_bit;

    amps.ptrs[hi] = cg.B.CreateConstGEP1_64(cg.vec_ty, ptr_sv_begin, idx_shift, "ptr.sv.hi");
    auto *amp_full = cg.B.CreateAlignedLoad(cg.vec_ty, amps.ptrs[hi],
                                            llvm::Align(cg.simd_width_bytes), "sv.full");

    for (unsigned li = 0; li < cg.LK; ++li) {
      amps.re[hi * cg.LK + li] = cg.B.CreateShuffleVector(
          amp_full, llvm::ArrayRef<int>(cg.smasks.re_split.data() + li * cg.s, cg.s), "re");
      amps.im[hi * cg.LK + li] = cg.B.CreateShuffleVector(
          amp_full, llvm::ArrayRef<int>(cg.smasks.im_split.data() + li * cg.s, cg.s), "im");
    }
  }

  return amps;
}

// ---------------------------------------------------------------------------
// Phase 2: Complex matrix-vector multiply
//
//   new_re[r] = Σ_c  re_mat[r,c] * re_amp[c]  −  im_mat[r,c] * im_amp[c]
//   new_im[r] = Σ_c  re_mat[r,c] * im_amp[c]  +  im_mat[r,c] * re_amp[c]
//
// Literal 0/±1 matrix entries are folded by InstCombine under fast-math;
// adjacent fmul+fadd pairs are contracted into FMAs by the backend.
// ---------------------------------------------------------------------------

struct MatvecResult {
  std::vector<llvm::Value *> re; // LK elements
  std::vector<llvm::Value *> im; // LK elements
};

static MatvecResult emit_matvec(KernelCodegen &cg, const LoadedAmplitudes &amps, unsigned hi) {
  auto *zero_vec =
      llvm::ConstantAggregateZero::get(llvm::VectorType::get(cg.scalar_ty, cg.s, false));

  MatvecResult result;
  result.re.resize(cg.LK, nullptr);
  result.im.resize(cg.LK, nullptr);

  for (unsigned li = 0; li < cg.LK; ++li) {
    const unsigned r = hi * cg.LK + li;
    for (unsigned c = 0; c < cg.K; ++c) {
      const auto &e = cg.mat_data[r * cg.K + c];

      auto *re_re = cg.B.CreateFMul(e.re_vec, amps.re[c], "re.re");
      auto *im_im = cg.B.CreateFMul(e.im_vec, amps.im[c], "im.im");
      auto *re_contrib = cg.B.CreateFSub(re_re, im_im, "re.contrib");
      result.re[li] =
          result.re[li] ? cg.B.CreateFAdd(result.re[li], re_contrib, "acc.re") : re_contrib;

      auto *re_im = cg.B.CreateFMul(e.re_vec, amps.im[c], "re.im");
      auto *im_re = cg.B.CreateFMul(e.im_vec, amps.re[c], "im.re");
      auto *im_contrib = cg.B.CreateFAdd(re_im, im_re, "im.contrib");
      result.im[li] =
          result.im[li] ? cg.B.CreateFAdd(result.im[li], im_contrib, "acc.im") : im_contrib;
    }
    if (result.re[li] == nullptr)
      result.re[li] = zero_vec;
    if (result.im[li] == nullptr)
      result.im[li] = zero_vec;
  }

  return result;
}

// ---------------------------------------------------------------------------
// Phase 2b: Blocked matrix-vector multiply.
//
// Same arithmetic as `emit_matvec`, but:
//   * Output rows are processed in groups of T (`CAST_BLOCK_GEMM_T`).
//   * Each block runs in its own basic block.
//   * Block accumulators are written back to a caller-allocated scratch
//     buffer via `store volatile`; the final BB reloads them via
//     `load volatile`.  The volatile ops are the barrier that actually
//     retires the accumulators between blocks — without them, SSA
//     continuity means LLVM's regalloc sees a single gigantic live range
//     regardless of how we structure the C++ emission loops.
//
// Register-pressure model: during block `bi`'s matvec inner loop, the
// peak live-set is (2·K amp vectors) + (2·T block accumulators) + O(1)
// matrix constants — vs the legacy path's (2·K amp vectors) + (2·K
// result accumulators).  At k=6, T=8 this halves peak pressure from
// 256 to 144 values.
//
// Caller contract: `re_scratch`/`im_scratch` are entry-block allocas of
// length LK × vec_s_ty.  This function leaves the IRBuilder's insertion
// point at the `done` BB.
// ---------------------------------------------------------------------------

static MatvecResult emit_matvec_blocked(KernelCodegen &cg, const LoadedAmplitudes &amps,
                                        unsigned hi, unsigned T, llvm::Value *re_scratch,
                                        llvm::Value *im_scratch) {
  assert(T > 0);
  assert(cg.LK > T && "caller must gate on K > T (only blocks when it helps)");

  auto *vec_s_ty = llvm::VectorType::get(cg.scalar_ty, cg.s, false);
  auto *zero_vec = llvm::ConstantAggregateZero::get(vec_s_ty);
  const auto align = llvm::Align(cg.simd_width_bytes);

  const unsigned n_blocks = (cg.LK + T - 1) / T;
  auto *func = cg.B.GetInsertBlock()->getParent();
  auto &ctx = func->getContext();

  // Pre-create block BBs + final BB so each block ends with a br to the
  // next.  SimplifyCFG may merge these in O1, but the volatile ops remain
  // in place — that's what retires the accumulators.
  std::vector<llvm::BasicBlock *> block_bbs(n_blocks + 1);
  for (unsigned i = 0; i < n_blocks; ++i) {
    block_bbs[i] = llvm::BasicBlock::Create(
        ctx, "matvec.blk." + std::to_string(hi) + "." + std::to_string(i), func);
  }
  block_bbs[n_blocks] =
      llvm::BasicBlock::Create(ctx, "matvec.done." + std::to_string(hi), func);

  cg.B.CreateBr(block_bbs[0]);

  for (unsigned bi = 0; bi < n_blocks; ++bi) {
    cg.B.SetInsertPoint(block_bbs[bi]);
    const unsigned r_start = bi * T;
    const unsigned r_end = std::min(r_start + T, cg.LK);
    const unsigned block_sz = r_end - r_start;

    std::vector<llvm::Value *> acc_re(block_sz, nullptr);
    std::vector<llvm::Value *> acc_im(block_sz, nullptr);

    // Inner c-loop iterates the full K columns, accumulating into this
    // block's T rows.  The ordering (c outer, ti inner) keeps amp[c] as
    // a short-lived reference within each iteration.
    for (unsigned c = 0; c < cg.K; ++c) {
      for (unsigned ti = 0; ti < block_sz; ++ti) {
        const unsigned li = r_start + ti;
        const unsigned r = hi * cg.LK + li;
        const auto &e = cg.mat_data[r * cg.K + c];

        auto *re_re = cg.B.CreateFMul(e.re_vec, amps.re[c], "re.re");
        auto *im_im = cg.B.CreateFMul(e.im_vec, amps.im[c], "im.im");
        auto *re_contrib = cg.B.CreateFSub(re_re, im_im, "re.contrib");
        acc_re[ti] = acc_re[ti] ? cg.B.CreateFAdd(acc_re[ti], re_contrib, "acc.re") : re_contrib;

        auto *re_im = cg.B.CreateFMul(e.re_vec, amps.im[c], "re.im");
        auto *im_re = cg.B.CreateFMul(e.im_vec, amps.re[c], "im.re");
        auto *im_contrib = cg.B.CreateFAdd(re_im, im_re, "im.contrib");
        acc_im[ti] = acc_im[ti] ? cg.B.CreateFAdd(acc_im[ti], im_contrib, "acc.im") : im_contrib;
      }
    }

    // Retire block accumulators to the scratch buffer via volatile stores.
    // Volatile forbids elision/reordering, so the SSA values' live ranges
    // end here.
    for (unsigned ti = 0; ti < block_sz; ++ti) {
      const unsigned li = r_start + ti;
      auto *v_re = acc_re[ti] ? acc_re[ti] : zero_vec;
      auto *v_im = acc_im[ti] ? acc_im[ti] : zero_vec;
      auto *p_re = cg.B.CreateConstGEP1_32(vec_s_ty, re_scratch, li, "p.re.scratch");
      auto *p_im = cg.B.CreateConstGEP1_32(vec_s_ty, im_scratch, li, "p.im.scratch");
      auto *st_re = cg.B.CreateAlignedStore(v_re, p_re, align);
      auto *st_im = cg.B.CreateAlignedStore(v_im, p_im, align);
      st_re->setVolatile(true);
      st_im->setVolatile(true);
    }

    cg.B.CreateBr(block_bbs[bi + 1]);
  }

  // Final BB: volatile-reload the full result array.
  cg.B.SetInsertPoint(block_bbs[n_blocks]);
  MatvecResult result;
  result.re.resize(cg.LK);
  result.im.resize(cg.LK);
  for (unsigned li = 0; li < cg.LK; ++li) {
    auto *p_re = cg.B.CreateConstGEP1_32(vec_s_ty, re_scratch, li, "p.re.load");
    auto *p_im = cg.B.CreateConstGEP1_32(vec_s_ty, im_scratch, li, "p.im.load");
    auto *v_re = cg.B.CreateAlignedLoad(vec_s_ty, p_re, align, "blk.re");
    auto *v_im = cg.B.CreateAlignedLoad(vec_s_ty, p_im, align, "blk.im");
    llvm::cast<llvm::LoadInst>(v_re)->setVolatile(true);
    llvm::cast<llvm::LoadInst>(v_im)->setVolatile(true);
    result.re[li] = v_re;
    result.im[li] = v_im;
  }
  return result;
}

// Allocates per-hi-block scratch space in the function's entry block.
// Returns {re_scratch, im_scratch}; both of type `[LK × <s × scalar>]*`.
// Call once per kernel, before `emit_loop_body`.
static std::pair<llvm::Value *, llvm::Value *>
alloca_matvec_scratch(KernelCodegen &cg, llvm::BasicBlock &entry_bb) {
  llvm::IRBuilder<> entry_builder(&entry_bb, entry_bb.getFirstInsertionPt());
  auto *vec_s_ty = llvm::VectorType::get(cg.scalar_ty, cg.s, false);
  auto *n_elems = entry_builder.getInt32(cg.LK);
  auto *re = entry_builder.CreateAlloca(vec_s_ty, n_elems, "mv.re.scratch");
  auto *im = entry_builder.CreateAlloca(vec_s_ty, n_elems, "mv.im.scratch");
  re->setAlignment(llvm::Align(cg.simd_width_bytes));
  im->setAlignment(llvm::Align(cg.simd_width_bytes));
  return {re, im};
}

// ---------------------------------------------------------------------------
// Phase 3: Merge lo-partitions and store
//
// Merge-sort the LK split vectors back into one via the precomputed masks,
// interleave re/im into [re0, im0, re1, im1, ...], and do an aligned store.
// ---------------------------------------------------------------------------

static void emit_merge_and_store(KernelCodegen &cg, MatvecResult &result, llvm::Value *p_sv_hi) {
  for (unsigned round = 0; round < cg.lk; ++round) {
    for (unsigned pair = 0; pair < (cg.LK >> round >> 1); ++pair) {
      const unsigned idx_l = pair << round << 1;
      const unsigned idx_r = idx_l | (1u << round);
      result.re[idx_l] = cg.B.CreateShuffleVector(result.re[idx_l], result.re[idx_r],
                                                  cg.smasks.merge[round], "re.merged");
      result.im[idx_l] = cg.B.CreateShuffleVector(result.im[idx_l], result.im[idx_r],
                                                  cg.smasks.merge[round], "im.merged");
    }
  }

  auto *merged =
      cg.B.CreateShuffleVector(result.re[0], result.im[0], cg.smasks.reim_merge, "amp.merged");
  cg.B.CreateAlignedStore(merged, p_sv_hi, llvm::Align(cg.simd_width_bytes));
}

// ---------------------------------------------------------------------------
// Emit the SV base pointer for a given task_id
// ---------------------------------------------------------------------------

static llvm::Value *emit_sv_base_ptr(KernelCodegen &cg, llvm::Value *p_sv, llvm::Value *task_id) {
  if (cg.layout.hi_bits.empty()) {
    return cg.B.CreateGEP(cg.vec_ty, p_sv, task_id, "ptr.sv.begin");
  }

  const auto segs = compute_hi_ptr_segments(cg.layout.hi_bits, cg.layout.sep_bit);
  llvm::Value *idx = cg.B.getInt64(0);
  for (const auto &seg : segs) {
    auto *part = cg.B.CreateAnd(task_id, seg.src_mask, "idx.part");
    if (seg.dst_shift > 0)
      part = cg.B.CreateShl(part, (uint64_t)seg.dst_shift, "idx.part");
    idx = cg.B.CreateAdd(idx, part, "idx");
  }
  return cg.B.CreateGEP(cg.vec_ty, p_sv, idx, "ptr.sv.begin");
}

// ---------------------------------------------------------------------------
// Tiled-load path for the all-lo case (hi_bits empty, HK=1).
//
// Replaces the single `<vec_size × scalar>` aligned load plus stride
// shufflevector extractions with `vec_size / s` aligned loads at native
// SIMD width, and the inverse on the store side.  This avoids LLVM's
// type-legalisation blowup for large vec_size (which explodes compile
// time at k≥5 when targets concentrate at low qubit positions).
//
// Correctness contract: the amp vector produced for (li, re/im) must have
// lane si equal to the statevector element at vector index
// re_split[li*s+si] (or im_split for im).  We decompose each such index
// into (chunk = idx >> simd_s, lane = idx & (s-1)) and use
// extractelement/insertelement on native-width chunks.  The store phase
// performs the inverse scatter.
// ---------------------------------------------------------------------------

static void emit_loop_body_tiled_all_lo(KernelCodegen &cg, llvm::Value *ptr_sv_begin) {
  assert(cg.layout.hi_bits.empty());
  assert(cg.HK == 1);
  const unsigned num_chunks = cg.vec_size / cg.s;
  assert(num_chunks >= 2);
  const unsigned s_mask = cg.s - 1;
  auto *chunk_ty = llvm::VectorType::get(cg.scalar_ty, cg.s, false);
  const auto align = llvm::Align(cg.simd_width_bytes);

  // 1. Load all native-width chunks in one pass.
  std::vector<llvm::Value *> chunks(num_chunks);
  for (unsigned c = 0; c < num_chunks; ++c) {
    auto *ptr = cg.B.CreateConstGEP1_32(chunk_ty, ptr_sv_begin, c, "ptr.chunk");
    chunks[c] = cg.B.CreateAlignedLoad(chunk_ty, ptr, align, "chunk");
  }

  // 2. Build amp vectors by gathering lanes from the chunks using the
  //    existing re_split / im_split index tables.
  auto *undef_vec = llvm::UndefValue::get(chunk_ty);
  LoadedAmplitudes amps;
  amps.re.resize(cg.LK);
  amps.im.resize(cg.LK);
  amps.ptrs.assign(1, ptr_sv_begin); // Unused for HK=1 (matvec only reads .re/.im).

  for (unsigned li = 0; li < cg.LK; ++li) {
    llvm::Value *re_vec = undef_vec;
    llvm::Value *im_vec = undef_vec;
    for (unsigned si = 0; si < cg.s; ++si) {
      const unsigned re_idx = static_cast<unsigned>(cg.smasks.re_split[li * cg.s + si]);
      const unsigned im_idx = static_cast<unsigned>(cg.smasks.im_split[li * cg.s + si]);
      auto *re_elem = cg.B.CreateExtractElement(chunks[re_idx >> cg.simd_s],
                                                uint64_t(re_idx & s_mask), "re.e");
      auto *im_elem = cg.B.CreateExtractElement(chunks[im_idx >> cg.simd_s],
                                                uint64_t(im_idx & s_mask), "im.e");
      re_vec = cg.B.CreateInsertElement(re_vec, re_elem, uint64_t(si), "re.v");
      im_vec = cg.B.CreateInsertElement(im_vec, im_elem, uint64_t(si), "im.v");
    }
    amps.re[li] = re_vec;
    amps.im[li] = im_vec;
  }

  // 3. Matvec: dispatch to blocked variant if the experiment is enabled
  //    and the gate is large enough to subdivide.
  auto result = (cg.block_gemm_t > 0 && cg.LK >= 4 * cg.block_gemm_t)
                    ? emit_matvec_blocked(cg, amps, /*hi=*/0, cg.block_gemm_t,
                                          cg.mv_re_scratch, cg.mv_im_scratch)
                    : emit_matvec(cg, amps, /*hi=*/0);

  // 4. Scatter result lanes back into native-width chunks.  Initial undef
  //    is fine: every lane of every chunk is reached exactly once by the
  //    nested loop below (LK·s = K = 2^k amps × s lanes = 2·LK·s lane
  //    writes total = 2^(k+simd_s+1) = vec_size = num_chunks·s, half re
  //    half im).
  std::vector<llvm::Value *> out_chunks(num_chunks, undef_vec);
  for (unsigned li = 0; li < cg.LK; ++li) {
    for (unsigned si = 0; si < cg.s; ++si) {
      const unsigned re_idx = static_cast<unsigned>(cg.smasks.re_split[li * cg.s + si]);
      const unsigned im_idx = static_cast<unsigned>(cg.smasks.im_split[li * cg.s + si]);
      auto *re_elem = cg.B.CreateExtractElement(result.re[li], uint64_t(si), "re.out.e");
      auto *im_elem = cg.B.CreateExtractElement(result.im[li], uint64_t(si), "im.out.e");
      const unsigned re_c = re_idx >> cg.simd_s;
      const unsigned im_c = im_idx >> cg.simd_s;
      out_chunks[re_c] = cg.B.CreateInsertElement(out_chunks[re_c], re_elem,
                                                  uint64_t(re_idx & s_mask), "out.re");
      out_chunks[im_c] = cg.B.CreateInsertElement(out_chunks[im_c], im_elem,
                                                  uint64_t(im_idx & s_mask), "out.im");
    }
  }

  for (unsigned c = 0; c < num_chunks; ++c) {
    auto *ptr = cg.B.CreateConstGEP1_32(chunk_ty, ptr_sv_begin, c, "ptr.out");
    cg.B.CreateAlignedStore(out_chunks[c], ptr, align);
  }
}

// ---------------------------------------------------------------------------
// Emit the full loop body: load → matvec → merge/store for each hi block
// ---------------------------------------------------------------------------

static void emit_loop_body(KernelCodegen &cg, llvm::Value *ptr_sv_begin) {
  // Tile only when all targets are lo_bits (HK=1 — with hi iterations the
  // per-hi load is already a native-width slice) AND the mega-load would
  // exceed one native-width vector (otherwise there is no legalisation to
  // avoid).
  //
  // At s=16 (F32 W512) the tiled path emits 2·s = 32 extract/insert pairs
  // per amp vector, which InstCombine converts to 16-wide shufflevectors;
  // native codegen of those outweighs the legalisation savings on this
  // single variant (empirically a net +50% at k=6).  Gate it out; all
  // other {precision, SIMD} combinations show a net win on the isolated
  // microbench.
  if (cg.layout.hi_bits.empty() && cg.vec_size > cg.s && cg.s <= 8) {
    emit_loop_body_tiled_all_lo(cg, ptr_sv_begin);
    return;
  }

  auto amps = emit_load_amplitudes(cg, ptr_sv_begin);

  for (unsigned hi = 0; hi < cg.HK; ++hi) {
    auto result = (cg.block_gemm_t > 0 && cg.LK >= 4 * cg.block_gemm_t)
                      ? emit_matvec_blocked(cg, amps, hi, cg.block_gemm_t,
                                            cg.mv_re_scratch, cg.mv_im_scratch)
                      : emit_matvec(cg, amps, hi);
    emit_merge_and_store(cg, result, amps.ptrs[hi]);
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

llvm::Expected<llvm::Function *> cast_cpu_generate_kernel_ir(
    const cast_cpu_kernel_gen_spec_t &spec, const cast_cpu_complex64_t *matrix, size_t matrix_len,
    const uint32_t *qubits, size_t n_qubits, llvm::StringRef func_name, llvm::Module &module) {

  // --- Validation ---
  if (!cast_cpu_detail::is_valid_precision(spec.precision))
    return llvm::createStringError("invalid precision");
  if (!cast_cpu_detail::is_valid_simd_width(spec.simd_width))
    return llvm::createStringError("invalid SIMD width");
  if (!cast_cpu_detail::is_valid_mode(spec.mode))
    return llvm::createStringError("invalid matrix load mode");
  if (matrix == nullptr)
    return llvm::createStringError("matrix must not be null");
  if (qubits == nullptr && n_qubits != 0)
    return llvm::createStringError("qubits must not be null");
  for (size_t i = 1; i < n_qubits; ++i) {
    if (qubits[i - 1] >= qubits[i])
      return llvm::createStringError("qubits must be strictly ascending");
  }
  size_t expected_len = 0;
  if (!cast_cpu_detail::expected_matrix_len(n_qubits, &expected_len) || expected_len != matrix_len)
    return llvm::createStringError("matrix length does not match the target qubit count");

  // --- Derived constants ---
  const unsigned simd_s = cast_cpu_detail::get_simd_s(spec.simd_width, spec.precision);
  assert(simd_s > 0 && simd_s <= 4);
  const unsigned s = 1u << simd_s;
  const unsigned k = static_cast<unsigned>(n_qubits);
  const unsigned K = 1u << k;
  const MatrixView mat_view{matrix, static_cast<uint32_t>(K)};

  const BitLayout layout = compute_bit_layout(qubits, n_qubits, simd_s);
  const unsigned lk = layout.lo_bits.size();
  const unsigned LK = 1u << lk;
  const unsigned hk = layout.hi_bits.size();
  const unsigned HK = 1u << hk;
  const unsigned vec_size = 1u << layout.sep_bit;
  const unsigned simd_width_bytes = static_cast<unsigned>(spec.simd_width) / 8u;

  const ShuffleMasks smasks = compute_shuffle_masks(layout, s, simd_s, vec_size);

  // --- IR setup ---
  auto &ctx = module.getContext();
  llvm::IRBuilder<> builder(ctx);
  builder.setFastMathFlags(llvm::FastMathFlags::getFast());

  auto *scalar_ty =
      (spec.precision == CAST_CPU_PRECISION_F32) ? builder.getFloatTy() : builder.getDoubleTy();
  auto *vec_ty = llvm::VectorType::get(scalar_ty, vec_size, false);

  // --- Create function skeleton ---
  auto *launch_ty = llvm::StructType::get(builder.getPtrTy(), builder.getInt64Ty(),
                                          builder.getInt64Ty(), builder.getPtrTy());
  auto *func_ty = llvm::FunctionType::get(builder.getVoidTy(), {builder.getPtrTy()}, false);
  auto *func = llvm::Function::Create(func_ty, llvm::Function::ExternalLinkage, func_name, module);

  auto *entry_bb = llvm::BasicBlock::Create(ctx, "entry", func);
  auto *loop_bb = llvm::BasicBlock::Create(ctx, "loop", func);
  auto *loop_body_bb = llvm::BasicBlock::Create(ctx, "loop.body", func);
  auto *ret_bb = llvm::BasicBlock::Create(ctx, "ret", func);

  // --- Entry block: unpack LaunchArgs and preload matrix ---
  LaunchArgs args;
  builder.SetInsertPoint(entry_bb);
  auto *launch_arg = func->getArg(0);
  args.p_sv = builder.CreateLoad(
      builder.getPtrTy(), builder.CreateStructGEP(launch_ty, launch_arg, 0, "launch.sv.ptr"), "sv");
  args.ctr_begin = builder.CreateLoad(
      builder.getInt64Ty(),
      builder.CreateStructGEP(launch_ty, launch_arg, 1, "launch.ctr.begin.ptr"), "ctr.begin");
  args.ctr_end = builder.CreateLoad(
      builder.getInt64Ty(), builder.CreateStructGEP(launch_ty, launch_arg, 2, "launch.ctr.end.ptr"),
      "ctr.end");
  args.p_mat = builder.CreateLoad(
      builder.getPtrTy(), builder.CreateStructGEP(launch_ty, launch_arg, 3, "launch.mat.ptr"),
      "mat");

  const auto mat_data = build_matrix_data(builder, spec, mat_view, args.p_mat, simd_s);

  // --- Block-GEMM experiment: read env var and, if active, allocate the
  //     matvec scratch buffer in the entry block (one allocation per
  //     kernel, reused across loop iterations).  Gated on `LK >= 4·T`:
  //     - LK=1 (hi-heavy) has one accumulator per hi iteration → blocking
  //       can't reduce pressure;
  //     - LK ≤ 2·T gives too few blocks to amortise the volatile-store
  //       overhead (empirically regresses k=4 runtime by ~8%);
  //     - LK ≥ 4·T is the empirical break-even (k=5 neutral, k=6 clear win).
  const unsigned block_gemm_t = get_block_gemm_tile();
  const bool block_gemm_active = block_gemm_t > 0 && LK >= 4 * block_gemm_t;

  // --- Construct codegen context ---
  KernelCodegen cg{
      builder,          k,      K,      lk,       LK,        hk,    HK, s, simd_s, vec_size,
      simd_width_bytes, layout, smasks, mat_data, scalar_ty, vec_ty, block_gemm_t};

  if (block_gemm_active) {
    auto scratch = alloca_matvec_scratch(cg, *entry_bb);
    cg.mv_re_scratch = scratch.first;
    cg.mv_im_scratch = scratch.second;
  }

  // --- Loop ---
  builder.CreateBr(loop_bb);
  builder.SetInsertPoint(loop_bb);
  auto *task_id = builder.CreatePHI(builder.getInt64Ty(), 2, "taskid");
  task_id->addIncoming(args.ctr_begin, entry_bb);
  builder.CreateCondBr(builder.CreateICmpSLT(task_id, args.ctr_end, "cond"), loop_body_bb, ret_bb);

  builder.SetInsertPoint(loop_body_bb);
  auto *ptr_sv_begin = emit_sv_base_ptr(cg, args.p_sv, task_id);
  emit_loop_body(cg, ptr_sv_begin);

  // emit_loop_body may have split the body across additional BBs (e.g.
  // block-GEMM's per-block BBs).  Use GetInsertBlock() to get the actual
  // predecessor of the back-edge.
  auto *task_id_next = builder.CreateAdd(task_id, builder.getInt64(1), "taskid.next");
  auto *loop_tail_bb = builder.GetInsertBlock();
  task_id->addIncoming(task_id_next, loop_tail_bb);
  builder.CreateBr(loop_bb);

  builder.SetInsertPoint(ret_bb);
  builder.CreateRetVoid();

  return func;
}
