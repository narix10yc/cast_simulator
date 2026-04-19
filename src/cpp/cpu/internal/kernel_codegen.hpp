#ifndef CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_KERNEL_CODEGEN_HPP
#define CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_KERNEL_CODEGEN_HPP

#include "bit_layout.hpp"
#include "matrix_data.hpp"
#include "shuffle_masks.hpp"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Alignment.h>

#include <vector>

namespace cast::cpu {

struct LoadedAmplitudes {
  std::vector<llvm::Value *> re;   // K = LK*HK elements, re[hi*LK + li]
  std::vector<llvm::Value *> im;   // K elements
  std::vector<llvm::Value *> ptrs; // HK pointers, used by Phase 3 stores
};

struct MatvecResult {
  std::vector<llvm::Value *> re; // LK elements
  std::vector<llvm::Value *> im;
};

// Block-mode volatile retire/reload scratch.  Null when Block mode isn't
// engaged; otherwise points at entry-block `[LK × <S × scalar>]` allocas.
struct MatvecScratch {
  llvm::Value *re = nullptr;
  llvm::Value *im = nullptr;
  bool engaged() const { return re != nullptr; }
};

// Straight — emitMatvec (default); Block — emitMatvecBlocked.
enum class MatvecMode { Straight, Block };

// Mega — one wide aligned load + shuffle; Tiled — native-width chunked loads.
enum class LoadMode { Mega, Tiled };

// Full kernel emission strategy, selected once per kernel by chooseStrategy().
struct KernelStrategy {
  LoadMode loadMode = LoadMode::Mega;
  MatvecMode matvecMode = MatvecMode::Straight;
  unsigned tileT = 0; // meaningful iff Block
};

KernelStrategy chooseStrategy(const BitLayout &layout, unsigned vecRegs, LoadMode loadMode);

// scalarTy = float or double; vecTy = <vecSize() × scalar>.
struct TypeBundle {
  llvm::Type *scalarTy;
  llvm::Type *vecTy;
};

// Per-kernel emission context.  BitLayout is the authoritative shape;
// size constants are read via its accessors.
struct KernelCodegen {
  llvm::IRBuilder<> &B;

  const BitLayout &layout;
  unsigned simdWidthBytes; // for aligned loads/stores
  const ShuffleMasks &smasks;
  const std::vector<IRMatData> &matData;
  const TypeBundle &types;

  KernelStrategy strategy;     // selected by chooseStrategy() in ctor
  MatvecScratch matvecScratch; // engaged() iff strategy.matvecMode == Block

  KernelCodegen(llvm::IRBuilder<> &builder, const BitLayout &layout, unsigned simdWidthBytes,
                const ShuffleMasks &smasks, const std::vector<IRMatData> &matData,
                const TypeBundle &types, unsigned vecRegs, LoadMode loadMode,
                llvm::BasicBlock &entryBb);

  // Native-width complex-lane vector type (`<S × scalar>`) and SIMD alignment.
  llvm::VectorType *vecSType() const {
    return llvm::VectorType::get(types.scalarTy, layout.S(), false);
  }
  llvm::Align simdAlign() const { return llvm::Align(simdWidthBytes); }

  // Allocate the Block-mode scratch buffers in `entryBb`.  Called once by
  // the ctor when Block is engaged; consumed by every emitMatvecBlocked.
  MatvecScratch allocaMatvecScratch(llvm::BasicBlock &entryBb) const;

  llvm::Value *emitSvBasePtr(llvm::Value *p_sv, llvm::Value *taskId);

  // Phase 1: load all K amplitudes.  Dispatches Mega/Tiled per partition.
  LoadedAmplitudes emitLoadAmplitudes(llvm::Value *ptrSvBegin);

  // Pointer to the start of hi-combination `hi` within the statevector segment.
  llvm::Value *computeHiPtr(llvm::Value *ptrSvBegin, unsigned hi);

  // Phase 2 — Straight: one LK·K accumulator chain; LLVM owns regalloc.
  MatvecResult emitMatvec(const LoadedAmplitudes &amps, unsigned hi);

  // Phase 2 — Block: same arithmetic, output rows tiled in T with volatile
  // retire/reload.  chooseStrategy() gates on `LK ≥ R` (≡ `LK ≥ 4·T`).
  MatvecResult emitMatvecBlocked(const LoadedAmplitudes &amps, unsigned hi, unsigned T,
                                 llvm::Value *reScratch, llvm::Value *imScratch);

  // Phase 2 — Straight/Block dispatcher; reads strategy.matvecMode.
  MatvecResult emitMatvecDispatched(const LoadedAmplitudes &amps, unsigned hi);

  // Block-mode helpers.  Volatility retires accumulators between blocks —
  // without it SimplifyCFG would collapse the per-block BBs.
  void retireBlockToScratch(unsigned rStart, unsigned blockSz,
                            const std::vector<llvm::Value *> &accRe,
                            const std::vector<llvm::Value *> &accIm, llvm::Value *reScratch,
                            llvm::Value *imScratch);
  MatvecResult reloadFullResultFromScratch(llvm::Value *reScratch, llvm::Value *imScratch);

  // Phase 3: Mega/Tiled store dispatcher.
  void emitStoreResult(MatvecResult &result, llvm::Value *ptr_hi);

  // Phase 3 Mega: merge lo-partitions, interleave re/im, aligned store.
  void emitMergeAndStore(MatvecResult &result, llvm::Value *p_sv_hi);

  void emitLoopBody(llvm::Value *ptrSvBegin);

  // Tiled load/store helpers.  `chunkTy` is `vecSType()`.
  std::vector<llvm::Value *> loadAllChunks(llvm::Value *ptrSvBegin, unsigned numChunks,
                                           llvm::VectorType *chunkTy);
  LoadedAmplitudes gatherAmpsFromChunks(const std::vector<llvm::Value *> &chunks,
                                        llvm::VectorType *chunkTy);
  std::vector<llvm::Value *> scatterResultIntoChunks(const MatvecResult &result, unsigned numChunks,
                                                     llvm::VectorType *chunkTy);
  void storeAllChunks(const std::vector<llvm::Value *> &outChunks, llvm::Value *ptrSvBegin,
                      llvm::VectorType *chunkTy);
};

} // namespace cast::cpu

#endif // CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_KERNEL_CODEGEN_HPP
