#ifndef CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_KERNEL_CODEGEN_H
#define CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_KERNEL_CODEGEN_H

#include "bit_layout.h"
#include "matrix_data.h"
#include "shuffle_masks.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Alignment.h>

#include <vector>

namespace cast_cpu_detail {

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

// Straight — emit_matvec (default); Block — emit_matvec_blocked (opt-in).
enum class MatvecMode { Straight, Block };

// scalar_ty = float or double; vec_ty = <vec_size() × scalar>.
struct TypeBundle {
  llvm::Type *scalar_ty;
  llvm::Type *vec_ty;
};

// Per-kernel emission context.  BitLayout is the authoritative shape;
// size constants are read via its accessors.
struct KernelCodegen {
  llvm::IRBuilder<> &B;

  const BitLayout &layout;
  unsigned simd_width_bytes; // for aligned loads/stores
  const ShuffleMasks &smasks;
  const std::vector<IRMatData> &mat_data;
  const TypeBundle &types;

  // `block_gemm_t == 0` → Straight for this kernel.  `> 0` is a request;
  // `matvec_mode` carries the final decision (gate `LK ≥ 4·T` applied in
  // the ctor and stored here; no site below re-derives it).
  unsigned block_gemm_t;
  MatvecMode matvec_mode;
  MatvecScratch matvec_scratch; // engaged() iff matvec_mode == Block

  KernelCodegen(llvm::IRBuilder<> &builder, const BitLayout &layout, unsigned simd_width_bytes,
                const ShuffleMasks &smasks, const std::vector<IRMatData> &mat_data,
                const TypeBundle &types, unsigned block_gemm_t, llvm::BasicBlock &entry_bb);

  // Native-width complex-lane vector type (`<S × scalar>`) and SIMD alignment.
  llvm::VectorType *vec_s_type() const {
    return llvm::VectorType::get(types.scalar_ty, layout.S(), false);
  }
  llvm::Align simd_align() const { return llvm::Align(simd_width_bytes); }

  // Allocate the Block-mode scratch buffers in `entry_bb`.  Called once by
  // the ctor when Block is engaged; consumed by every emit_matvec_blocked.
  MatvecScratch alloca_matvec_scratch(llvm::BasicBlock &entry_bb) const;

  llvm::Value *emit_sv_base_ptr(llvm::Value *p_sv, llvm::Value *task_id);

  // Phase 1: per-hi aligned load + shuffle-split into re/im.
  LoadedAmplitudes emit_load_amplitudes(llvm::Value *ptr_sv_begin);

  // Phase 2 — Straight: one LK·K accumulator chain; LLVM owns regalloc.
  MatvecResult emit_matvec(const LoadedAmplitudes &amps, unsigned hi);

  // Phase 2 — Block: same arithmetic, output rows tiled in T with volatile
  // retire/reload.  Caller gates `LK > T`; dispatch sites enforce `LK ≥ 4·T`.
  MatvecResult emit_matvec_blocked(const LoadedAmplitudes &amps, unsigned hi, unsigned T,
                                   llvm::Value *re_scratch, llvm::Value *im_scratch);

  // Phase 2 — Straight/Block dispatcher; the only consumer of matvec_mode.
  MatvecResult emit_matvec_dispatched(const LoadedAmplitudes &amps, unsigned hi);

  // Block-mode helpers.  Volatility retires accumulators between blocks —
  // without it SimplifyCFG would collapse the per-block BBs.
  void retire_block_to_scratch(unsigned r_start, unsigned block_sz,
                               const std::vector<llvm::Value *> &acc_re,
                               const std::vector<llvm::Value *> &acc_im, llvm::Value *re_scratch,
                               llvm::Value *im_scratch);
  MatvecResult reload_full_result_from_scratch(llvm::Value *re_scratch, llvm::Value *im_scratch);

  // Phase 3: merge lo-partitions, interleave re/im, aligned store.
  void emit_merge_and_store(MatvecResult &result, llvm::Value *p_sv_hi);

  void emit_loop_body_tiled_all_lo(llvm::Value *ptr_sv_begin);
  void emit_loop_body(llvm::Value *ptr_sv_begin); // dispatches tiled vs general

  // Tiled-all-lo helpers.  `chunk_ty` is `vec_s_type()`.
  std::vector<llvm::Value *> load_all_chunks(llvm::Value *ptr_sv_begin, unsigned num_chunks,
                                             llvm::VectorType *chunk_ty);
  LoadedAmplitudes gather_amps_from_chunks(const std::vector<llvm::Value *> &chunks,
                                           llvm::VectorType *chunk_ty);
  std::vector<llvm::Value *>
  scatter_result_into_chunks(const MatvecResult &result, unsigned num_chunks,
                             llvm::VectorType *chunk_ty);
  void store_all_chunks(const std::vector<llvm::Value *> &out_chunks, llvm::Value *ptr_sv_begin,
                        llvm::VectorType *chunk_ty);
};

} // namespace cast_cpu_detail

#endif // CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_KERNEL_CODEGEN_H
