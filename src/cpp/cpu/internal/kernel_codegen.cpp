#include "kernel_codegen.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Casting.h>

#include <algorithm>
#include <cassert>
#include <string>

namespace cast_cpu_detail {

KernelStrategy choose_strategy(const BitLayout &layout, unsigned vec_regs) {
  KernelStrategy s{};
  const unsigned T = std::max(2u, vec_regs / 4);

  if (layout.LK() >= vec_regs) {
    s.matvec_mode = MatvecMode::Block;
    s.tile_T = T;
  }

  bool all_lo = layout.hi_bits.empty();
  bool needs_legalize = all_lo && layout.vec_size() > layout.S();
  bool shuffle_cheap = layout.S() <= 8;
  if (needs_legalize && shuffle_cheap)
    s.load_mode = LoadMode::Tiled;

  return s;
}

KernelCodegen::KernelCodegen(llvm::IRBuilder<> &builder, const BitLayout &layout,
                             unsigned simd_width_bytes, const ShuffleMasks &smasks,
                             const std::vector<IRMatData> &mat_data, const TypeBundle &types,
                             unsigned vec_regs, llvm::BasicBlock &entry_bb)
    : B(builder), layout(layout), simd_width_bytes(simd_width_bytes), smasks(smasks),
      mat_data(mat_data), types(types), strategy(choose_strategy(layout, vec_regs)) {
  if (strategy.matvec_mode == MatvecMode::Block) {
    matvec_scratch = alloca_matvec_scratch(entry_bb);
  }
}

// Pointer to the start of hi-combination `hi` within the statevector segment.
// Returns `ptr_sv_begin` unchanged when hi == 0 (or when there are no hi bits).
llvm::Value *KernelCodegen::compute_hi_ptr(llvm::Value *ptr_sv_begin, unsigned hi) {
  uint64_t idx_shift = 0;
  for (unsigned hbit = 0; hbit < layout.hk(); ++hbit) {
    if (hi & (1u << hbit))
      idx_shift += uint64_t(1) << layout.hi_bits[hbit];
  }
  idx_shift >>= layout.sep_bit;
  if (idx_shift == 0)
    return ptr_sv_begin;
  return B.CreateConstGEP1_64(types.vec_ty, ptr_sv_begin, idx_shift);
}

// Phase 1 — load all K amplitudes across HK hi-combinations.
// Mega: one wide aligned load + ShuffleVector split per partition.
// Tiled: native-width chunk loads + scalar gather per partition.
LoadedAmplitudes KernelCodegen::emit_load_amplitudes(llvm::Value *ptr_sv_begin) {
  const auto LK = layout.LK();
  const auto HK = layout.HK();
  const auto S = layout.S();

  LoadedAmplitudes amps;
  amps.re.resize(layout.K());
  amps.im.resize(layout.K());
  amps.ptrs.resize(HK);

  for (unsigned hi = 0; hi < HK; ++hi) {
    amps.ptrs[hi] = compute_hi_ptr(ptr_sv_begin, hi);

    if (strategy.load_mode == LoadMode::Tiled) {
      auto *chunk_ty = vec_s_type();
      const unsigned num_chunks = layout.vec_size() / S;
      auto chunks = load_all_chunks(amps.ptrs[hi], num_chunks, chunk_ty);
      auto part = gather_amps_from_chunks(chunks, chunk_ty);
      std::copy(part.re.begin(), part.re.end(), amps.re.begin() + hi * LK);
      std::copy(part.im.begin(), part.im.end(), amps.im.begin() + hi * LK);
    } else {
      auto *amp_full = B.CreateAlignedLoad(types.vec_ty, amps.ptrs[hi], simd_align());
      for (unsigned li = 0; li < LK; ++li) {
        amps.re[hi * LK + li] = B.CreateShuffleVector(
            amp_full, llvm::ArrayRef<int>(smasks.re_split.data() + li * S, S));
        amps.im[hi * LK + li] = B.CreateShuffleVector(
            amp_full, llvm::ArrayRef<int>(smasks.im_split.data() + li * S, S));
      }
    }
  }

  return amps;
}

// Phase 2 Straight — straight-line SSA tree:
//   new_re[r] = Σ_c  re_mat[r,c] * re_amp[c] − im_mat[r,c] * im_amp[c]
//   new_im[r] = Σ_c  re_mat[r,c] * im_amp[c] + im_mat[r,c] * re_amp[c]
// InstCombine folds 0/±1 matrix entries; backend contracts into FMAs.  At
// k ≥ 5 the peak live set (~2·K vectors) overflows the ZMM file and LLVM
// spills heavily — that's what Block mode exists to mitigate.
MatvecResult KernelCodegen::emit_matvec(const LoadedAmplitudes &amps, unsigned hi) {
  const auto K = layout.K();
  const auto LK = layout.LK();
  assert(K > 0 && "emit_matvec requires K >= 1 (first c-iteration seeds the accumulators)");

  MatvecResult result;
  result.re.resize(LK, nullptr);
  result.im.resize(LK, nullptr);

  for (unsigned li = 0; li < LK; ++li) {
    const unsigned r = hi * LK + li;
    for (unsigned c = 0; c < K; ++c) {
      const auto &e = mat_data[r * K + c];

      auto *re_re = B.CreateFMul(e.re_vec, amps.re[c]);
      auto *im_im = B.CreateFMul(e.im_vec, amps.im[c]);
      auto *re_contrib = B.CreateFSub(re_re, im_im);
      result.re[li] = c == 0 ? re_contrib : B.CreateFAdd(result.re[li], re_contrib);

      auto *re_im = B.CreateFMul(e.re_vec, amps.im[c]);
      auto *im_re = B.CreateFMul(e.im_vec, amps.re[c]);
      auto *im_contrib = B.CreateFAdd(re_im, im_re);
      result.im[li] = c == 0 ? im_contrib : B.CreateFAdd(result.im[li], im_contrib);
    }
  }

  return result;
}

// Phase 2 Block — same arithmetic, output rows tiled in T.  Each block runs
// in its own BB; accumulators retire to scratch via `store volatile` and
// reload via `load volatile`.  Volatile ops are the barrier — without them
// SimplifyCFG collapses the per-block BBs into one gigantic live range.
//
// Live-set per block: (2·K amps) + (2·T block accs) + O(1) matrix consts,
// vs Straight's (2·K amps) + (2·K accs).  At k=6 T=8 this halves peak
// pressure from 256 to 144 values.
//
// Caller contract: `re_scratch`/`im_scratch` are `alloca_matvec_scratch`
// outputs; choose_strategy() gates on `LK ≥ vec_regs` (≡ `LK ≥ 4·T`,
// which amortizes the volatile-op overhead — see commit cd510df).  On
// return the builder is in `matvec.done`; returned values are the reloads.
MatvecResult KernelCodegen::emit_matvec_blocked(const LoadedAmplitudes &amps, unsigned hi,
                                                unsigned T, llvm::Value *re_scratch,
                                                llvm::Value *im_scratch) {
  const auto K = layout.K();
  const auto LK = layout.LK();
  assert(T > 0);
  assert(LK > T && "caller must gate on K > T (only blocks when it helps)");

  const unsigned n_blocks = (LK + T - 1) / T;
  auto *func = B.GetInsertBlock()->getParent();
  auto &ctx = func->getContext();

  // Pre-create block BBs + final BB so each block ends with a br to the
  // next.  SimplifyCFG may merge these in O1, but the volatile ops remain
  // in place — that's what retires the accumulators.
  std::vector<llvm::BasicBlock *> block_bbs(n_blocks + 1);
  for (unsigned i = 0; i < n_blocks; ++i) {
    block_bbs[i] = llvm::BasicBlock::Create(
        ctx, "matvec.blk." + std::to_string(hi) + "." + std::to_string(i), func);
  }
  block_bbs[n_blocks] = llvm::BasicBlock::Create(ctx, "matvec.done." + std::to_string(hi), func);

  B.CreateBr(block_bbs[0]);

  for (unsigned bi = 0; bi < n_blocks; ++bi) {
    B.SetInsertPoint(block_bbs[bi]);
    const unsigned r_start = bi * T;
    const unsigned r_end = std::min(r_start + T, LK);
    const unsigned block_sz = r_end - r_start;

    std::vector<llvm::Value *> acc_re(block_sz, nullptr);
    std::vector<llvm::Value *> acc_im(block_sz, nullptr);

    // c outer / ti inner keeps amp[c] short-lived per iteration.  c==0 seeds
    // the accumulators; later iterations FAdd into them.
    for (unsigned c = 0; c < K; ++c) {
      for (unsigned ti = 0; ti < block_sz; ++ti) {
        const unsigned li = r_start + ti;
        const unsigned r = hi * LK + li;
        const auto &e = mat_data[r * K + c];

        auto *re_re = B.CreateFMul(e.re_vec, amps.re[c]);
        auto *im_im = B.CreateFMul(e.im_vec, amps.im[c]);
        auto *re_contrib = B.CreateFSub(re_re, im_im);
        acc_re[ti] = c == 0 ? re_contrib : B.CreateFAdd(acc_re[ti], re_contrib);

        auto *re_im = B.CreateFMul(e.re_vec, amps.im[c]);
        auto *im_re = B.CreateFMul(e.im_vec, amps.re[c]);
        auto *im_contrib = B.CreateFAdd(re_im, im_re);
        acc_im[ti] = c == 0 ? im_contrib : B.CreateFAdd(acc_im[ti], im_contrib);
      }
    }

    retire_block_to_scratch(r_start, block_sz, acc_re, acc_im, re_scratch, im_scratch);
    B.CreateBr(block_bbs[bi + 1]);
  }

  B.SetInsertPoint(block_bbs[n_blocks]);
  return reload_full_result_from_scratch(re_scratch, im_scratch);
}

void KernelCodegen::retire_block_to_scratch(unsigned r_start, unsigned block_sz,
                                            const std::vector<llvm::Value *> &acc_re,
                                            const std::vector<llvm::Value *> &acc_im,
                                            llvm::Value *re_scratch, llvm::Value *im_scratch) {
  auto *vec_s_ty = vec_s_type();
  const auto align = simd_align();

  for (unsigned ti = 0; ti < block_sz; ++ti) {
    const unsigned li = r_start + ti;
    assert(acc_re[ti] != nullptr && acc_im[ti] != nullptr);
    auto *p_re = B.CreateConstGEP1_32(vec_s_ty, re_scratch, li);
    auto *p_im = B.CreateConstGEP1_32(vec_s_ty, im_scratch, li);
    auto *st_re = B.CreateAlignedStore(acc_re[ti], p_re, align);
    auto *st_im = B.CreateAlignedStore(acc_im[ti], p_im, align);
    st_re->setVolatile(true);
    st_im->setVolatile(true);
  }
}

MatvecResult KernelCodegen::reload_full_result_from_scratch(llvm::Value *re_scratch,
                                                            llvm::Value *im_scratch) {
  const auto LK = layout.LK();
  auto *vec_s_ty = vec_s_type();
  const auto align = simd_align();

  MatvecResult result;
  result.re.resize(LK);
  result.im.resize(LK);
  for (unsigned li = 0; li < LK; ++li) {
    auto *p_re = B.CreateConstGEP1_32(vec_s_ty, re_scratch, li);
    auto *p_im = B.CreateConstGEP1_32(vec_s_ty, im_scratch, li);
    auto *v_re = B.CreateAlignedLoad(vec_s_ty, p_re, align);
    auto *v_im = B.CreateAlignedLoad(vec_s_ty, p_im, align);
    llvm::cast<llvm::LoadInst>(v_re)->setVolatile(true);
    llvm::cast<llvm::LoadInst>(v_im)->setVolatile(true);
    result.re[li] = v_re;
    result.im[li] = v_im;
  }
  return result;
}

MatvecResult KernelCodegen::emit_matvec_dispatched(const LoadedAmplitudes &amps, unsigned hi) {
  if (strategy.matvec_mode == MatvecMode::Block) {
    return emit_matvec_blocked(amps, hi, strategy.tile_T, matvec_scratch.re, matvec_scratch.im);
  }
  return emit_matvec(amps, hi);
}

MatvecScratch KernelCodegen::alloca_matvec_scratch(llvm::BasicBlock &entry_bb) const {
  llvm::IRBuilder<> entry_builder(&entry_bb, entry_bb.getFirstInsertionPt());
  auto *vec_s_ty = vec_s_type();
  auto *n_elems = entry_builder.getInt32(layout.LK());
  auto *re = entry_builder.CreateAlloca(vec_s_ty, n_elems);
  auto *im = entry_builder.CreateAlloca(vec_s_ty, n_elems);
  re->setAlignment(simd_align());
  im->setAlignment(simd_align());
  return {re, im};
}

// Phase 3 — merge-sort LK split vectors back into one, interleave re/im as
// [re0, im0, re1, im1, ...], and aligned-store.
void KernelCodegen::emit_merge_and_store(MatvecResult &result, llvm::Value *p_sv_hi) {
  const auto LK = layout.LK();
  for (unsigned round = 0; round < layout.lk(); ++round) {
    for (unsigned pair = 0; pair < (LK >> round >> 1); ++pair) {
      const unsigned idx_l = pair << round << 1;
      const unsigned idx_r = idx_l | (1u << round);
      result.re[idx_l] =
          B.CreateShuffleVector(result.re[idx_l], result.re[idx_r], smasks.merge[round]);
      result.im[idx_l] =
          B.CreateShuffleVector(result.im[idx_l], result.im[idx_r], smasks.merge[round]);
    }
  }

  auto *merged = B.CreateShuffleVector(result.re[0], result.im[0], smasks.reim_merge);
  B.CreateAlignedStore(merged, p_sv_hi, simd_align());
}

// Phase 3 — Mega/Tiled store dispatcher.
void KernelCodegen::emit_store_result(MatvecResult &result, llvm::Value *ptr_hi) {
  if (strategy.load_mode == LoadMode::Tiled) {
    auto *chunk_ty = vec_s_type();
    const unsigned num_chunks = layout.vec_size() / layout.S();
    auto out_chunks = scatter_result_into_chunks(result, num_chunks, chunk_ty);
    store_all_chunks(out_chunks, ptr_hi, chunk_ty);
  } else {
    emit_merge_and_store(result, ptr_hi);
  }
}

llvm::Value *KernelCodegen::emit_sv_base_ptr(llvm::Value *p_sv, llvm::Value *task_id) {
  if (layout.hi_bits.empty()) {
    return B.CreateGEP(types.vec_ty, p_sv, task_id);
  }

  const auto segs = compute_hi_ptr_segments(layout.hi_bits, layout.sep_bit);
  llvm::Value *idx = B.getInt64(0);
  for (const auto &seg : segs) {
    auto *part = B.CreateAnd(task_id, seg.src_mask);
    if (seg.dst_shift > 0)
      part = B.CreateShl(part, (uint64_t)seg.dst_shift);
    idx = B.CreateAdd(idx, part);
  }
  return B.CreateGEP(types.vec_ty, p_sv, idx);
}

std::vector<llvm::Value *> KernelCodegen::load_all_chunks(llvm::Value *ptr_sv_begin,
                                                          unsigned num_chunks,
                                                          llvm::VectorType *chunk_ty) {
  const auto align = simd_align();
  std::vector<llvm::Value *> chunks(num_chunks);
  for (unsigned c = 0; c < num_chunks; ++c) {
    auto *ptr = B.CreateConstGEP1_32(chunk_ty, ptr_sv_begin, c);
    chunks[c] = B.CreateAlignedLoad(chunk_ty, ptr, align);
  }
  return chunks;
}

LoadedAmplitudes KernelCodegen::gather_amps_from_chunks(const std::vector<llvm::Value *> &chunks,
                                                        llvm::VectorType *chunk_ty) {
  const auto LK = layout.LK();
  const auto S = layout.S();
  const auto s = layout.s();
  const unsigned s_mask = S - 1;
  auto *poison_vec = llvm::PoisonValue::get(chunk_ty);

  LoadedAmplitudes amps;
  amps.re.resize(LK);
  amps.im.resize(LK);

  for (unsigned li = 0; li < LK; ++li) {
    llvm::Value *re_vec = poison_vec;
    llvm::Value *im_vec = poison_vec;
    for (unsigned si = 0; si < S; ++si) {
      const auto re_idx = static_cast<unsigned>(smasks.re_split[li * S + si]);
      const auto im_idx = static_cast<unsigned>(smasks.im_split[li * S + si]);
      auto *re_elem = B.CreateExtractElement(chunks[re_idx >> s], uint64_t(re_idx & s_mask));
      auto *im_elem = B.CreateExtractElement(chunks[im_idx >> s], uint64_t(im_idx & s_mask));
      re_vec = B.CreateInsertElement(re_vec, re_elem, uint64_t(si));
      im_vec = B.CreateInsertElement(im_vec, im_elem, uint64_t(si));
    }
    amps.re[li] = re_vec;
    amps.im[li] = im_vec;
  }
  return amps;
}

// Inverse of gather_amps_from_chunks.  Poison-init of chunks is safe: every
// lane is written exactly once (2·LK·S writes = vec_size).
std::vector<llvm::Value *> KernelCodegen::scatter_result_into_chunks(const MatvecResult &result,
                                                                     unsigned num_chunks,
                                                                     llvm::VectorType *chunk_ty) {
  const auto LK = layout.LK();
  const auto S = layout.S();
  const auto s = layout.s();
  const unsigned s_mask = S - 1;
  auto *poison_vec = llvm::PoisonValue::get(chunk_ty);

  std::vector<llvm::Value *> out_chunks(num_chunks, poison_vec);
  for (unsigned li = 0; li < LK; ++li) {
    for (unsigned si = 0; si < S; ++si) {
      const auto re_idx = static_cast<unsigned>(smasks.re_split[li * S + si]);
      const auto im_idx = static_cast<unsigned>(smasks.im_split[li * S + si]);
      auto *re_elem = B.CreateExtractElement(result.re[li], uint64_t(si));
      auto *im_elem = B.CreateExtractElement(result.im[li], uint64_t(si));
      out_chunks[re_idx >> s] =
          B.CreateInsertElement(out_chunks[re_idx >> s], re_elem, uint64_t(re_idx & s_mask));
      out_chunks[im_idx >> s] =
          B.CreateInsertElement(out_chunks[im_idx >> s], im_elem, uint64_t(im_idx & s_mask));
    }
  }
  return out_chunks;
}

void KernelCodegen::store_all_chunks(const std::vector<llvm::Value *> &out_chunks,
                                     llvm::Value *ptr_sv_begin, llvm::VectorType *chunk_ty) {
  const auto align = simd_align();
  for (unsigned c = 0; c < out_chunks.size(); ++c) {
    auto *ptr = B.CreateConstGEP1_32(chunk_ty, ptr_sv_begin, c);
    B.CreateAlignedStore(out_chunks[c], ptr, align);
  }
}

// Full loop body: load → matvec → store.  LoadMode and MatvecMode are
// selected by choose_strategy() and stored in `strategy`.
void KernelCodegen::emit_loop_body(llvm::Value *ptr_sv_begin) {
  auto amps = emit_load_amplitudes(ptr_sv_begin);
  for (unsigned hi = 0; hi < layout.HK(); ++hi) {
    auto result = emit_matvec_dispatched(amps, hi);
    emit_store_result(result, amps.ptrs[hi]);
  }
}

} // namespace cast_cpu_detail
