#include "cpu_gen.h"

#include "cpu_util.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>

#include <cassert>
#include <cstdint>
#include <vector>

namespace {

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

// One re/im vector pair per matrix element.  Zero, one, and minus-one entries
// are emitted as literal constants; the fast-math optimizer folds them away.
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

  // Adjust all positions for the implicit zero bit inserted at simd_s.
  for (auto *vec : {&layout.lo_bits, &layout.simd_bits, &layout.hi_bits}) {
    for (auto &bit : *vec) {
      if (bit >= simd_s)
        ++bit;
    }
  }

  layout.sep_bit = layout.simd_bits.empty() ? 0u : layout.simd_bits.back() + 1u;
  // If sep_bit lands on the inserted zero-bit position, bump past it.
  if (layout.sep_bit == simd_s)
    ++layout.sep_bit;

  return layout;
}

// ---------------------------------------------------------------------------
// Statevector base-pointer computation
//
// Each task_id indexes the "free" dimensions of the statevector (those not
// occupied by hi_bits).  The actual vec_ty* pointer is obtained by scattering
// task_id bits into the non-hi positions — i.e., inserting a 0 at each hi_bit
// position in the vec_ty index space.
//
// We precompute a list of (src_mask, dst_shift) segments.  Each segment
// extracts a contiguous run of task_id bits and places them at the
// corresponding free positions in the vec_ty index.  dst_shift equals the
// number of hi_bits encountered before that run (always >= 0).
// ---------------------------------------------------------------------------

struct PtrSegment {
  uint64_t src_mask;
  unsigned dst_shift;
};

static std::vector<PtrSegment> compute_hi_ptr_segments(const std::vector<unsigned> &hi_bits,
                                                       unsigned sep_bit) {
  std::vector<PtrSegment> segs;
  unsigned src_bit = 0;  // next unassigned bit in task_id
  unsigned prev_end = 0; // next free position in vec_ty index space

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

  // Remaining task_id bits above all hi_bit positions.
  segs.push_back({~((uint64_t(1) << src_bit) - 1), prev_end - src_bit});
  return segs;
}

static llvm::Value *emit_sv_base_ptr(llvm::IRBuilder<> &builder, llvm::Value *p_sv,
                                     llvm::Value *task_id, llvm::Type *vec_ty,
                                     const BitLayout &layout) {
  if (layout.hi_bits.empty()) {
    return builder.CreateGEP(vec_ty, p_sv, task_id, "ptr.sv.begin");
  }

  const auto segs = compute_hi_ptr_segments(layout.hi_bits, layout.sep_bit);
  llvm::Value *idx = builder.getInt64(0);
  for (const auto &seg : segs) {
    auto *part = builder.CreateAnd(task_id, seg.src_mask, "idx.part");
    if (seg.dst_shift > 0)
      part = builder.CreateShl(part, (uint64_t)seg.dst_shift, "idx.part");
    idx = builder.CreateAdd(idx, part, "idx");
  }
  return builder.CreateGEP(vec_ty, p_sv, idx, "ptr.sv.begin");
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
//
// re_split / im_split  — extract real and imaginary lanes from the interleaved
//                        [re0, im0, re1, im1, ...] SIMD vector into separate
//                        re / im vectors for each lo-qubit partition (li).
//
// merge[round]         — merge-sort shuffle mask for round `round` that
//                        reassembles pairs of updated amplitude vectors back
//                        into a single wider vector.  Built iteratively: each
//                        round merges two sorted index lists into one.
//
// reim_merge           — final interleave mask that packs the merged re/im
//                        results back into [re0, im0, re1, im1, ...] order.
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

  // --- Split masks ---
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

  // --- Merge masks ---
  // Each round merges two vectors of sorted indices into one, producing the
  // shuffle permutation for CreateShuffleVector.
  {
    std::vector<int> current(out.re_split.begin(), out.re_split.begin() + s);
    for (size_t round = 0; round < layout.lo_bits.size(); ++round) {
      const int half = static_cast<int>(current.size());

      // The right-hand group has the same indices as the left but with the
      // next lo_bit set — matching the amplitude partition at that lo_bit = 1.
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

  // --- Re-im interleave mask ---
  // Packs separate re/im vectors back into [re0, im0, re1, im1, ...] layout.
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
// Matrix data
//
// Each element is emitted as a splatted vector of its literal value (for
// IMM_VALUE mode) or a runtime load+splat (for STACK_LOAD mode).  Zeros,
// ones, and minus-ones are written as-is; fast-math folding in the optimizer
// (InstCombine + DAGCombine) eliminates the redundant multiplications.
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
  const unsigned s = 1u << simd_s; // SIMD lanes
  const unsigned k = static_cast<unsigned>(n_qubits);
  const unsigned K = 1u << k; // matrix dimension
  const MatrixView mat_view{matrix, static_cast<uint32_t>(K)};
  const unsigned simd_width_bytes = static_cast<unsigned>(spec.simd_width) / 8u;

  const BitLayout layout = compute_bit_layout(qubits, n_qubits, simd_s);
  const unsigned lk = layout.lo_bits.size();
  const unsigned LK = 1u << lk;
  const unsigned hk = layout.hi_bits.size();
  const unsigned HK = 1u << hk;
  const unsigned vec_size = 1u << layout.sep_bit;

  const ShuffleMasks smasks = compute_shuffle_masks(layout, s, simd_s, vec_size);

  // --- IR setup ---
  auto &ctx = module.getContext();
  llvm::IRBuilder<> builder(ctx);

  // Fast-math enables FMA contraction, constant folding of 0/±1 multiplies,
  // and reassociation — eliminating the need to special-case matrix elements.
  llvm::FastMathFlags fmf;
  fmf.setFast();
  builder.setFastMathFlags(fmf);

  auto *scalar_ty =
      (spec.precision == CAST_CPU_PRECISION_F32) ? builder.getFloatTy() : builder.getDoubleTy();
  auto *vec_ty = llvm::VectorType::get(scalar_ty, vec_size, false);

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

  // --- Loop header ---
  builder.CreateBr(loop_bb);
  builder.SetInsertPoint(loop_bb);
  auto *task_id = builder.CreatePHI(builder.getInt64Ty(), 2, "taskid");
  task_id->addIncoming(args.ctr_begin, entry_bb);
  builder.CreateCondBr(builder.CreateICmpSLT(task_id, args.ctr_end, "cond"), loop_body_bb, ret_bb);

  // --- Loop body ---
  builder.SetInsertPoint(loop_body_bb);

  auto *ptr_sv_begin = emit_sv_base_ptr(builder, args.p_sv, task_id, vec_ty, layout);

  // Load amplitudes: for each hi-qubit combination, do one aligned load and
  // shuffle into separate re/im vectors for each lo-qubit partition.
  std::vector<llvm::Value *> re_amps(K);
  std::vector<llvm::Value *> im_amps(K);
  std::vector<llvm::Value *> p_svs(HK);
  for (unsigned hi = 0; hi < HK; ++hi) {
    uint64_t idx_shift = 0;
    for (unsigned hbit = 0; hbit < hk; ++hbit) {
      if (hi & (1u << hbit))
        idx_shift += uint64_t(1) << layout.hi_bits[hbit];
    }
    idx_shift >>= layout.sep_bit;
    p_svs[hi] = builder.CreateConstGEP1_64(vec_ty, ptr_sv_begin, idx_shift, "ptr.sv.hi");
    auto *amp_full =
        builder.CreateAlignedLoad(vec_ty, p_svs[hi], llvm::Align(simd_width_bytes), "sv.full");
    for (unsigned li = 0; li < LK; ++li) {
      re_amps[hi * LK + li] = builder.CreateShuffleVector(
          amp_full, llvm::ArrayRef<int>(smasks.re_split.data() + li * s, s), "re");
      im_amps[hi * LK + li] = builder.CreateShuffleVector(
          amp_full, llvm::ArrayRef<int>(smasks.im_split.data() + li * s, s), "im");
    }
  }

  // Complex matrix–vector multiply:
  //   new_re[r] = Σ_c  re_mat[r,c] * re_amp[c]  −  im_mat[r,c] * im_amp[c]
  //   new_im[r] = Σ_c  re_mat[r,c] * im_amp[c]  +  im_mat[r,c] * re_amp[c]
  //
  // Literal 0/±1 matrix entries are folded by InstCombine under fast-math;
  // adjacent fmul+fadd pairs are contracted into FMAs by the backend.
  auto *zero_vec = llvm::ConstantAggregateZero::get(llvm::VectorType::get(scalar_ty, s, false));
  std::vector<llvm::Value *> updated_re(LK);
  std::vector<llvm::Value *> updated_im(LK);
  for (unsigned hi = 0; hi < HK; ++hi) {
    std::fill(updated_re.begin(), updated_re.end(), nullptr);
    std::fill(updated_im.begin(), updated_im.end(), nullptr);

    for (unsigned li = 0; li < LK; ++li) {
      const unsigned r = hi * LK + li;
      for (unsigned c = 0; c < K; ++c) {
        const auto &e = mat_data[r * K + c];

        auto *re_re = builder.CreateFMul(e.re_vec, re_amps[c], "re.re");
        auto *im_im = builder.CreateFMul(e.im_vec, im_amps[c], "im.im");
        auto *re_contrib = builder.CreateFSub(re_re, im_im, "re.contrib");
        updated_re[li] =
            updated_re[li] ? builder.CreateFAdd(updated_re[li], re_contrib, "acc.re") : re_contrib;

        auto *re_im = builder.CreateFMul(e.re_vec, im_amps[c], "re.im");
        auto *im_re = builder.CreateFMul(e.im_vec, re_amps[c], "im.re");
        auto *im_contrib = builder.CreateFAdd(re_im, im_re, "im.contrib");
        updated_im[li] =
            updated_im[li] ? builder.CreateFAdd(updated_im[li], im_contrib, "acc.im") : im_contrib;
      }
      if (updated_re[li] == nullptr)
        updated_re[li] = zero_vec;
      if (updated_im[li] == nullptr)
        updated_im[li] = zero_vec;
    }

    // Merge LK split vectors back into one via the precomputed merge-sort masks.
    for (unsigned round = 0; round < lk; ++round) {
      for (unsigned pair = 0; pair < (LK >> round >> 1); ++pair) {
        const unsigned idx_l = pair << round << 1;
        const unsigned idx_r = idx_l | (1u << round);
        updated_re[idx_l] = builder.CreateShuffleVector(updated_re[idx_l], updated_re[idx_r],
                                                        smasks.merge[round], "re.merged");
        updated_im[idx_l] = builder.CreateShuffleVector(updated_im[idx_l], updated_im[idx_r],
                                                        smasks.merge[round], "im.merged");
      }
    }

    // Interleave re/im back into [re0, im0, re1, im1, ...] and store.
    auto *merged =
        builder.CreateShuffleVector(updated_re[0], updated_im[0], smasks.reim_merge, "amp.merged");
    builder.CreateAlignedStore(merged, p_svs[hi], llvm::Align(simd_width_bytes));
  }

  auto *task_id_next = builder.CreateAdd(task_id, builder.getInt64(1), "taskid.next");
  task_id->addIncoming(task_id_next, loop_body_bb);
  builder.CreateBr(loop_bb);

  builder.SetInsertPoint(ret_bb);
  builder.CreateRetVoid();

  return func;
}
