#include "kernel_codegen.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Casting.h>

#include <algorithm>
#include <cassert>
#include <string>

namespace cast::cpu {

KernelStrategy chooseStrategy(const BitLayout &layout, unsigned vecRegs, LoadMode loadMode) {
  KernelStrategy s{};
  const unsigned T = std::max(2u, vecRegs / 4);

  if (layout.LK() >= vecRegs) {
    s.matvecMode = MatvecMode::Block;
    s.tileT = T;
  }

  s.loadMode = loadMode;

  return s;
}

KernelCodegen::KernelCodegen(llvm::IRBuilder<> &builder, const BitLayout &layout,
                             unsigned simdWidthBytes, const ShuffleMasks &smasks,
                             const std::vector<IRMatData> &matData, const TypeBundle &types,
                             unsigned vecRegs, LoadMode loadMode, llvm::BasicBlock &entryBb)
    : B(builder), layout(layout), simdWidthBytes(simdWidthBytes), smasks(smasks), matData(matData),
      types(types), strategy(chooseStrategy(layout, vecRegs, loadMode)) {
  if (strategy.matvecMode == MatvecMode::Block) {
    matvecScratch = allocaMatvecScratch(entryBb);
  }
}

// Pointer to the start of hi-combination `hi` within the statevector segment.
// Returns `ptrSvBegin` unchanged when hi == 0 (or when there are no hi bits).
llvm::Value *KernelCodegen::computeHiPtr(llvm::Value *ptrSvBegin, unsigned hi) {
  uint64_t idxShift = 0;
  for (unsigned hbit = 0; hbit < layout.hk(); ++hbit) {
    if (hi & (1u << hbit))
      idxShift += uint64_t(1) << layout.hiBits[hbit];
  }
  idxShift >>= layout.sepBit;
  if (idxShift == 0)
    return ptrSvBegin;
  return B.CreateConstGEP1_64(types.vecTy, ptrSvBegin, idxShift);
}

// Phase 1 — load all K amplitudes across HK hi-combinations.
// Mega: one wide aligned load + ShuffleVector split per partition.
// Tiled: native-width chunk loads + scalar gather per partition.
LoadedAmplitudes KernelCodegen::emitLoadAmplitudes(llvm::Value *ptrSvBegin) {
  const auto LK = layout.LK();
  const auto HK = layout.HK();
  const auto S = layout.S();

  LoadedAmplitudes amps;
  amps.re.resize(layout.K());
  amps.im.resize(layout.K());
  amps.ptrs.resize(HK);

  for (unsigned hi = 0; hi < HK; ++hi) {
    amps.ptrs[hi] = computeHiPtr(ptrSvBegin, hi);

    if (strategy.loadMode == LoadMode::Tiled) {
      auto *chunkTy = vecSType();
      const unsigned numChunks = layout.vecSize() / S;
      auto chunks = loadAllChunks(amps.ptrs[hi], numChunks, chunkTy);
      auto part = gatherAmpsFromChunks(chunks, chunkTy);
      std::copy(part.re.begin(), part.re.end(), amps.re.begin() + hi * LK);
      std::copy(part.im.begin(), part.im.end(), amps.im.begin() + hi * LK);
    } else {
      auto *ampFull = B.CreateAlignedLoad(types.vecTy, amps.ptrs[hi], simdAlign());
      for (unsigned li = 0; li < LK; ++li) {
        amps.re[hi * LK + li] =
            B.CreateShuffleVector(ampFull, llvm::ArrayRef<int>(smasks.reSplit.data() + li * S, S));
        amps.im[hi * LK + li] =
            B.CreateShuffleVector(ampFull, llvm::ArrayRef<int>(smasks.imSplit.data() + li * S, S));
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
MatvecResult KernelCodegen::emitMatvec(const LoadedAmplitudes &amps, unsigned hi) {
  const auto K = layout.K();
  const auto LK = layout.LK();
  assert(K > 0 && "emitMatvec requires K >= 1 (first c-iteration seeds the accumulators)");

  MatvecResult result;
  result.re.resize(LK, nullptr);
  result.im.resize(LK, nullptr);

  for (unsigned li = 0; li < LK; ++li) {
    const unsigned r = hi * LK + li;
    for (unsigned c = 0; c < K; ++c) {
      const auto &e = matData[r * K + c];

      auto *reRe = B.CreateFMul(e.reVec, amps.re[c]);
      auto *imIm = B.CreateFMul(e.imVec, amps.im[c]);
      auto *reContrib = B.CreateFSub(reRe, imIm);
      result.re[li] = c == 0 ? reContrib : B.CreateFAdd(result.re[li], reContrib);

      auto *reIm = B.CreateFMul(e.reVec, amps.im[c]);
      auto *imRe = B.CreateFMul(e.imVec, amps.re[c]);
      auto *imContrib = B.CreateFAdd(reIm, imRe);
      result.im[li] = c == 0 ? imContrib : B.CreateFAdd(result.im[li], imContrib);
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
// Caller contract: `reScratch`/`imScratch` are `allocaMatvecScratch`
// outputs; chooseStrategy() gates on `LK ≥ vecRegs` (≡ `LK ≥ 4·T`,
// which amortizes the volatile-op overhead — see commit cd510df).  On
// return the builder is in `matvec.done`; returned values are the reloads.
MatvecResult KernelCodegen::emitMatvecBlocked(const LoadedAmplitudes &amps, unsigned hi, unsigned T,
                                              llvm::Value *reScratch, llvm::Value *imScratch) {
  const auto K = layout.K();
  const auto LK = layout.LK();
  assert(T > 0);
  assert(LK > T && "caller must gate on K > T (only blocks when it helps)");

  const unsigned nBlocks = (LK + T - 1) / T;
  auto *func = B.GetInsertBlock()->getParent();
  auto &ctx = func->getContext();

  // Pre-create block BBs + final BB so each block ends with a br to the
  // next.  SimplifyCFG may merge these in O1, but the volatile ops remain
  // in place — that's what retires the accumulators.
  std::vector<llvm::BasicBlock *> blockBbs(nBlocks + 1);
  for (unsigned i = 0; i < nBlocks; ++i) {
    blockBbs[i] = llvm::BasicBlock::Create(
        ctx, "matvec.blk." + std::to_string(hi) + "." + std::to_string(i), func);
  }
  blockBbs[nBlocks] = llvm::BasicBlock::Create(ctx, "matvec.done." + std::to_string(hi), func);

  B.CreateBr(blockBbs[0]);

  for (unsigned bi = 0; bi < nBlocks; ++bi) {
    B.SetInsertPoint(blockBbs[bi]);
    const unsigned rStart = bi * T;
    const unsigned rEnd = std::min(rStart + T, LK);
    const unsigned blockSz = rEnd - rStart;

    std::vector<llvm::Value *> accRe(blockSz, nullptr);
    std::vector<llvm::Value *> accIm(blockSz, nullptr);

    // c outer / ti inner keeps amp[c] short-lived per iteration.  c==0 seeds
    // the accumulators; later iterations FAdd into them.
    for (unsigned c = 0; c < K; ++c) {
      for (unsigned ti = 0; ti < blockSz; ++ti) {
        const unsigned li = rStart + ti;
        const unsigned r = hi * LK + li;
        const auto &e = matData[r * K + c];

        auto *reRe = B.CreateFMul(e.reVec, amps.re[c]);
        auto *imIm = B.CreateFMul(e.imVec, amps.im[c]);
        auto *reContrib = B.CreateFSub(reRe, imIm);
        accRe[ti] = c == 0 ? reContrib : B.CreateFAdd(accRe[ti], reContrib);

        auto *reIm = B.CreateFMul(e.reVec, amps.im[c]);
        auto *imRe = B.CreateFMul(e.imVec, amps.re[c]);
        auto *imContrib = B.CreateFAdd(reIm, imRe);
        accIm[ti] = c == 0 ? imContrib : B.CreateFAdd(accIm[ti], imContrib);
      }
    }

    retireBlockToScratch(rStart, blockSz, accRe, accIm, reScratch, imScratch);
    B.CreateBr(blockBbs[bi + 1]);
  }

  B.SetInsertPoint(blockBbs[nBlocks]);
  return reloadFullResultFromScratch(reScratch, imScratch);
}

void KernelCodegen::retireBlockToScratch(unsigned rStart, unsigned blockSz,
                                         const std::vector<llvm::Value *> &accRe,
                                         const std::vector<llvm::Value *> &accIm,
                                         llvm::Value *reScratch, llvm::Value *imScratch) {
  auto *vecSTy = vecSType();
  const auto align = simdAlign();

  for (unsigned ti = 0; ti < blockSz; ++ti) {
    const unsigned li = rStart + ti;
    assert(accRe[ti] != nullptr && accIm[ti] != nullptr);
    auto *pRe = B.CreateConstGEP1_32(vecSTy, reScratch, li);
    auto *pIm = B.CreateConstGEP1_32(vecSTy, imScratch, li);
    auto *stRe = B.CreateAlignedStore(accRe[ti], pRe, align);
    auto *stIm = B.CreateAlignedStore(accIm[ti], pIm, align);
    stRe->setVolatile(true);
    stIm->setVolatile(true);
  }
}

MatvecResult KernelCodegen::reloadFullResultFromScratch(llvm::Value *reScratch,
                                                        llvm::Value *imScratch) {
  const auto LK = layout.LK();
  auto *vecSTy = vecSType();
  const auto align = simdAlign();

  MatvecResult result;
  result.re.resize(LK);
  result.im.resize(LK);
  for (unsigned li = 0; li < LK; ++li) {
    auto *pRe = B.CreateConstGEP1_32(vecSTy, reScratch, li);
    auto *pIm = B.CreateConstGEP1_32(vecSTy, imScratch, li);
    auto *vRe = B.CreateAlignedLoad(vecSTy, pRe, align);
    auto *vIm = B.CreateAlignedLoad(vecSTy, pIm, align);
    llvm::cast<llvm::LoadInst>(vRe)->setVolatile(true);
    llvm::cast<llvm::LoadInst>(vIm)->setVolatile(true);
    result.re[li] = vRe;
    result.im[li] = vIm;
  }
  return result;
}

MatvecResult KernelCodegen::emitMatvecDispatched(const LoadedAmplitudes &amps, unsigned hi) {
  if (strategy.matvecMode == MatvecMode::Block) {
    return emitMatvecBlocked(amps, hi, strategy.tileT, matvecScratch.re, matvecScratch.im);
  }
  return emitMatvec(amps, hi);
}

MatvecScratch KernelCodegen::allocaMatvecScratch(llvm::BasicBlock &entryBb) const {
  llvm::IRBuilder<> entryBuilder(&entryBb, entryBb.getFirstInsertionPt());
  auto *vecSTy = vecSType();
  auto *nElems = entryBuilder.getInt32(layout.LK());
  auto *re = entryBuilder.CreateAlloca(vecSTy, nElems);
  auto *im = entryBuilder.CreateAlloca(vecSTy, nElems);
  re->setAlignment(simdAlign());
  im->setAlignment(simdAlign());
  return {re, im};
}

// Phase 3 — merge-sort LK split vectors back into one, interleave re/im as
// [re0, im0, re1, im1, ...], and aligned-store.
void KernelCodegen::emitMergeAndStore(MatvecResult &result, llvm::Value *p_sv_hi) {
  const auto LK = layout.LK();
  for (unsigned round = 0; round < layout.lk(); ++round) {
    for (unsigned pair = 0; pair < (LK >> round >> 1); ++pair) {
      const unsigned idxL = pair << round << 1;
      const unsigned idxR = idxL | (1u << round);
      result.re[idxL] =
          B.CreateShuffleVector(result.re[idxL], result.re[idxR], smasks.merge[round]);
      result.im[idxL] =
          B.CreateShuffleVector(result.im[idxL], result.im[idxR], smasks.merge[round]);
    }
  }

  auto *merged = B.CreateShuffleVector(result.re[0], result.im[0], smasks.reimMerge);
  B.CreateAlignedStore(merged, p_sv_hi, simdAlign());
}

// Phase 3 — Mega/Tiled store dispatcher.
void KernelCodegen::emitStoreResult(MatvecResult &result, llvm::Value *ptr_hi) {
  if (strategy.loadMode == LoadMode::Tiled) {
    auto *chunkTy = vecSType();
    const unsigned numChunks = layout.vecSize() / layout.S();
    auto outChunks = scatterResultIntoChunks(result, numChunks, chunkTy);
    storeAllChunks(outChunks, ptr_hi, chunkTy);
  } else {
    emitMergeAndStore(result, ptr_hi);
  }
}

llvm::Value *KernelCodegen::emitSvBasePtr(llvm::Value *p_sv, llvm::Value *taskId) {
  if (layout.hiBits.empty()) {
    return B.CreateGEP(types.vecTy, p_sv, taskId);
  }

  const auto segs = computeHiPtrSegments(layout.hiBits, layout.sepBit);
  llvm::Value *idx = B.getInt64(0);
  for (const auto &seg : segs) {
    auto *part = B.CreateAnd(taskId, seg.srcMask);
    if (seg.dstShift > 0)
      part = B.CreateShl(part, (uint64_t)seg.dstShift);
    idx = B.CreateAdd(idx, part);
  }
  return B.CreateGEP(types.vecTy, p_sv, idx);
}

std::vector<llvm::Value *> KernelCodegen::loadAllChunks(llvm::Value *ptrSvBegin, unsigned numChunks,
                                                        llvm::VectorType *chunkTy) {
  const auto align = simdAlign();
  std::vector<llvm::Value *> chunks(numChunks);
  for (unsigned c = 0; c < numChunks; ++c) {
    auto *ptr = B.CreateConstGEP1_32(chunkTy, ptrSvBegin, c);
    chunks[c] = B.CreateAlignedLoad(chunkTy, ptr, align);
  }
  return chunks;
}

LoadedAmplitudes KernelCodegen::gatherAmpsFromChunks(const std::vector<llvm::Value *> &chunks,
                                                     llvm::VectorType *chunkTy) {
  const auto LK = layout.LK();
  const auto S = layout.S();
  const auto s = layout.s();
  const unsigned sMask = S - 1;
  auto *poisonVec = llvm::PoisonValue::get(chunkTy);

  LoadedAmplitudes amps;
  amps.re.resize(LK);
  amps.im.resize(LK);

  for (unsigned li = 0; li < LK; ++li) {
    llvm::Value *reVec = poisonVec;
    llvm::Value *imVec = poisonVec;
    for (unsigned si = 0; si < S; ++si) {
      const auto reIdx = static_cast<unsigned>(smasks.reSplit[li * S + si]);
      const auto imIdx = static_cast<unsigned>(smasks.imSplit[li * S + si]);
      auto *reElem = B.CreateExtractElement(chunks[reIdx >> s], uint64_t(reIdx & sMask));
      auto *imElem = B.CreateExtractElement(chunks[imIdx >> s], uint64_t(imIdx & sMask));
      reVec = B.CreateInsertElement(reVec, reElem, uint64_t(si));
      imVec = B.CreateInsertElement(imVec, imElem, uint64_t(si));
    }
    amps.re[li] = reVec;
    amps.im[li] = imVec;
  }
  return amps;
}

// Inverse of gatherAmpsFromChunks.  Poison-init of chunks is safe: every
// lane is written exactly once (2·LK·S writes = vecSize).
std::vector<llvm::Value *> KernelCodegen::scatterResultIntoChunks(const MatvecResult &result,
                                                                  unsigned numChunks,
                                                                  llvm::VectorType *chunkTy) {
  const auto LK = layout.LK();
  const auto S = layout.S();
  const auto s = layout.s();
  const unsigned sMask = S - 1;
  auto *poisonVec = llvm::PoisonValue::get(chunkTy);

  std::vector<llvm::Value *> outChunks(numChunks, poisonVec);
  for (unsigned li = 0; li < LK; ++li) {
    for (unsigned si = 0; si < S; ++si) {
      const auto reIdx = static_cast<unsigned>(smasks.reSplit[li * S + si]);
      const auto imIdx = static_cast<unsigned>(smasks.imSplit[li * S + si]);
      auto *reElem = B.CreateExtractElement(result.re[li], uint64_t(si));
      auto *imElem = B.CreateExtractElement(result.im[li], uint64_t(si));
      outChunks[reIdx >> s] =
          B.CreateInsertElement(outChunks[reIdx >> s], reElem, uint64_t(reIdx & sMask));
      outChunks[imIdx >> s] =
          B.CreateInsertElement(outChunks[imIdx >> s], imElem, uint64_t(imIdx & sMask));
    }
  }
  return outChunks;
}

void KernelCodegen::storeAllChunks(const std::vector<llvm::Value *> &outChunks,
                                   llvm::Value *ptrSvBegin, llvm::VectorType *chunkTy) {
  const auto align = simdAlign();
  for (unsigned c = 0; c < outChunks.size(); ++c) {
    auto *ptr = B.CreateConstGEP1_32(chunkTy, ptrSvBegin, c);
    B.CreateAlignedStore(outChunks[c], ptr, align);
  }
}

// Full loop body: load → matvec → store.  LoadMode and MatvecMode are
// selected by chooseStrategy() and stored in `strategy`.
void KernelCodegen::emitLoopBody(llvm::Value *ptrSvBegin) {
  auto amps = emitLoadAmplitudes(ptrSvBegin);
  for (unsigned hi = 0; hi < layout.HK(); ++hi) {
    auto result = emitMatvecDispatched(amps, hi);
    emitStoreResult(result, amps.ptrs[hi]);
  }
}

} // namespace cast::cpu
