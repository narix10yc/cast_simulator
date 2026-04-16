#include "cpu_gen.h"
#include "internal/bit_layout.h"
#include "internal/kernel_codegen.h"
#include "internal/matrix_data.h"
#include "internal/shuffle_masks.h"
#include "internal/util.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/FMF.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace {

using cast_cpu_detail::BitLayout;
using cast_cpu_detail::build_matrix_data;
using cast_cpu_detail::compute_bit_layout;
using cast_cpu_detail::compute_shuffle_masks;
using cast_cpu_detail::KernelCodegen;
using cast_cpu_detail::MatrixView;
using cast_cpu_detail::ShuffleMasks;
using cast_cpu_detail::TypeBundle;

// Live state unpacked from the opaque launch-args struct by the entry BB.
struct LaunchArgs {
  llvm::Value *p_sv = nullptr;
  llvm::Value *ctr_begin = nullptr;
  llvm::Value *ctr_end = nullptr;
  llvm::Value *p_mat = nullptr;
};

// Pre-created function + BBs + LaunchArgs struct type for one kernel.
struct FunctionSkeleton {
  llvm::Function *func;
  llvm::BasicBlock *entry_bb;
  llvm::BasicBlock *loop_bb;
  llvm::BasicBlock *loop_body_bb;
  llvm::BasicBlock *ret_bb;
  llvm::StructType *launch_ty;
};

llvm::Error validate_kernel_gen_inputs(const cast_cpu_kernel_gen_spec_t &spec,
                                       const cast_cpu_complex64_t *matrix, size_t matrix_len,
                                       const uint32_t *qubits, size_t n_qubits) {
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
  return llvm::Error::success();
}

// Creates the function + entry/loop/loop.body/ret BBs.  Does not set the
// builder's insertion point.
FunctionSkeleton create_function_skeleton(llvm::Module &module, llvm::IRBuilder<> &builder,
                                          llvm::StringRef func_name) {
  auto &ctx = module.getContext();
  auto *launch_ty = llvm::StructType::get(builder.getPtrTy(), builder.getInt64Ty(),
                                          builder.getInt64Ty(), builder.getPtrTy());
  auto *func_ty = llvm::FunctionType::get(builder.getVoidTy(), {builder.getPtrTy()}, false);
  auto *func = llvm::Function::Create(func_ty, llvm::Function::ExternalLinkage, func_name, module);

  return {func,
          llvm::BasicBlock::Create(ctx, "entry", func),
          llvm::BasicBlock::Create(ctx, "loop", func),
          llvm::BasicBlock::Create(ctx, "loop.body", func),
          llvm::BasicBlock::Create(ctx, "ret", func),
          launch_ty};
}

// Caller must set the builder's insertion point to entry_bb first.
LaunchArgs unpack_launch_args(llvm::IRBuilder<> &builder, llvm::Function *func,
                              llvm::StructType *launch_ty) {
  LaunchArgs args;
  auto *launch_arg = func->getArg(0);
  args.p_sv =
      builder.CreateLoad(builder.getPtrTy(), builder.CreateStructGEP(launch_ty, launch_arg, 0));
  args.ctr_begin =
      builder.CreateLoad(builder.getInt64Ty(), builder.CreateStructGEP(launch_ty, launch_arg, 1));
  args.ctr_end =
      builder.CreateLoad(builder.getInt64Ty(), builder.CreateStructGEP(launch_ty, launch_arg, 2));
  args.p_mat =
      builder.CreateLoad(builder.getPtrTy(), builder.CreateStructGEP(launch_ty, launch_arg, 3));
  return args;
}

// PHI-indexed task_id loop.  Back-edge uses GetInsertBlock() because Block
// mode may have split the body.  Leaves the builder at ret_bb.
void emit_taskid_loop(llvm::IRBuilder<> &builder, KernelCodegen &cg, const LaunchArgs &args,
                      const FunctionSkeleton &skel) {
  builder.CreateBr(skel.loop_bb);
  builder.SetInsertPoint(skel.loop_bb);
  auto *task_id = builder.CreatePHI(builder.getInt64Ty(), 2);
  task_id->addIncoming(args.ctr_begin, skel.entry_bb);
  builder.CreateCondBr(builder.CreateICmpSLT(task_id, args.ctr_end), skel.loop_body_bb,
                       skel.ret_bb);

  builder.SetInsertPoint(skel.loop_body_bb);
  auto *ptr_sv_begin = cg.emit_sv_base_ptr(args.p_sv, task_id);
  cg.emit_loop_body(ptr_sv_begin);

  auto *task_id_next = builder.CreateAdd(task_id, builder.getInt64(1));
  auto *loop_tail_bb = builder.GetInsertBlock();
  task_id->addIncoming(task_id_next, loop_tail_bb);
  builder.CreateBr(skel.loop_bb);

  builder.SetInsertPoint(skel.ret_bb);
  builder.CreateRetVoid();
}

} // namespace

llvm::Expected<llvm::Function *> cast_cpu_generate_kernel_ir(
    const cast_cpu_kernel_gen_spec_t &spec, const cast_cpu_complex64_t *matrix, size_t matrix_len,
    const uint32_t *qubits, size_t n_qubits, llvm::StringRef func_name, llvm::Module &module) {
  if (auto err = validate_kernel_gen_inputs(spec, matrix, matrix_len, qubits, n_qubits))
    return std::move(err);

  const unsigned s = cast_cpu_detail::get_simd_s(spec.simd_width, spec.precision);
  assert(s > 0 && s <= 4);
  const MatrixView mat_view{matrix, static_cast<uint32_t>(1u << n_qubits)};

  const BitLayout layout = compute_bit_layout(qubits, n_qubits, s);
  const unsigned simd_width_bytes = static_cast<unsigned>(spec.simd_width) / 8u;
  const ShuffleMasks smasks =
      compute_shuffle_masks(layout, layout.S(), layout.s(), layout.vec_size());

  llvm::IRBuilder<> builder(module.getContext());
  builder.setFastMathFlags(llvm::FastMathFlags::getFast());
  auto *scalar_ty =
      (spec.precision == CAST_CPU_PRECISION_F32) ? builder.getFloatTy() : builder.getDoubleTy();
  const TypeBundle types{scalar_ty, llvm::VectorType::get(scalar_ty, layout.vec_size(), false)};

  const FunctionSkeleton skel = create_function_skeleton(module, builder, func_name);
  builder.SetInsertPoint(skel.entry_bb);
  const LaunchArgs args = unpack_launch_args(builder, skel.func, skel.launch_ty);
  const auto mat_data = build_matrix_data(builder, spec, mat_view, args.p_mat, scalar_ty, s);

  // Default 32 (matches AVX-512, NEON, SVE).  Override with CAST_VEC_REGS for
  // A/B benchmarking (e.g. 9999 → force Straight on all gates).
  static const unsigned vec_regs = [] {
    const char *s = std::getenv("CAST_VEC_REGS");
    if (s && s[0] != '\0') {
      char *end = nullptr;
      const long v = std::strtol(s, &end, 10);
      if (end && *end == '\0' && v >= 2)
        return static_cast<unsigned>(v);
    }
    return 32u;
  }();
  KernelCodegen cg(builder, layout, simd_width_bytes, smasks, mat_data, types, vec_regs,
                   *skel.entry_bb);

  emit_taskid_loop(builder, cg, args, skel);
  return skel.func;
}
