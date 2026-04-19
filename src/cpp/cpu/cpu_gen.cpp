#include "cpu_gen.hpp"
#include "internal/bit_layout.hpp"
#include "internal/kernel_codegen.hpp"
#include "internal/matrix_data.hpp"
#include "internal/shuffle_masks.hpp"
#include "internal/util.hpp"

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

using cast::cpu::BitLayout;
using cast::cpu::buildMatrixData;
using cast::cpu::computeBitLayout;
using cast::cpu::computeShuffleMasks;
using cast::cpu::KernelCodegen;
using cast::cpu::MatrixView;
using cast::cpu::ShuffleMasks;
using cast::cpu::TypeBundle;

// Live state unpacked from the opaque launch-args struct by the entry BB.
struct LaunchArgs {
  llvm::Value *p_sv = nullptr;
  llvm::Value *ctrBegin = nullptr;
  llvm::Value *ctrEnd = nullptr;
  llvm::Value *p_mat = nullptr;
};

// Pre-created function + BBs + LaunchArgs struct type for one kernel.
struct FunctionSkeleton {
  llvm::Function *func;
  llvm::BasicBlock *entryBb;
  llvm::BasicBlock *loopBb;
  llvm::BasicBlock *loopBodyBb;
  llvm::BasicBlock *retBb;
  llvm::StructType *launchTy;
};

llvm::Error validateKernelGenInputs(const cast::cpu::KernelGenSpec &spec,
                                    const cast::Complex64 *matrix, size_t matrixLen,
                                    const uint32_t *qubits, size_t nQubits) {
  if (!cast::isValidPrecision(spec.precision))
    return llvm::createStringError("invalid precision");
  if (!cast::cpu::isValidSimdWidth(spec.simdWidth))
    return llvm::createStringError("invalid SIMD width");
  if (!cast::cpu::isValidMode(spec.mode))
    return llvm::createStringError("invalid matrix load mode");
  if (matrix == nullptr)
    return llvm::createStringError("matrix must not be null");
  if (qubits == nullptr && nQubits != 0)
    return llvm::createStringError("qubits must not be null");
  for (size_t i = 1; i < nQubits; ++i) {
    if (qubits[i - 1] >= qubits[i])
      return llvm::createStringError("qubits must be strictly ascending");
  }
  size_t expectedLen = 0;
  if (!cast::cpu::expectedMatrixLen(nQubits, &expectedLen) || expectedLen != matrixLen)
    return llvm::createStringError("matrix length does not match the target qubit count");
  return llvm::Error::success();
}

// Creates the function + entry/loop/loop.body/ret BBs.  Does not set the
// builder's insertion point.
FunctionSkeleton createFunctionSkeleton(llvm::Module &module, llvm::IRBuilder<> &builder,
                                        llvm::StringRef funcName) {
  auto &ctx = module.getContext();
  auto *launchTy = llvm::StructType::get(builder.getPtrTy(), builder.getInt64Ty(),
                                         builder.getInt64Ty(), builder.getPtrTy());
  auto *funcTy = llvm::FunctionType::get(builder.getVoidTy(), {builder.getPtrTy()}, false);
  auto *func = llvm::Function::Create(funcTy, llvm::Function::ExternalLinkage, funcName, module);

  return {func,
          llvm::BasicBlock::Create(ctx, "entry", func),
          llvm::BasicBlock::Create(ctx, "loop", func),
          llvm::BasicBlock::Create(ctx, "loop.body", func),
          llvm::BasicBlock::Create(ctx, "ret", func),
          launchTy};
}

// Caller must set the builder's insertion point to entryBb first.
LaunchArgs unpackLaunchArgs(llvm::IRBuilder<> &builder, llvm::Function *func,
                            llvm::StructType *launchTy) {
  LaunchArgs args;
  auto *launchArg = func->getArg(0);
  args.p_sv =
      builder.CreateLoad(builder.getPtrTy(), builder.CreateStructGEP(launchTy, launchArg, 0));
  args.ctrBegin =
      builder.CreateLoad(builder.getInt64Ty(), builder.CreateStructGEP(launchTy, launchArg, 1));
  args.ctrEnd =
      builder.CreateLoad(builder.getInt64Ty(), builder.CreateStructGEP(launchTy, launchArg, 2));
  args.p_mat =
      builder.CreateLoad(builder.getPtrTy(), builder.CreateStructGEP(launchTy, launchArg, 3));
  return args;
}

// PHI-indexed taskId loop.  Back-edge uses GetInsertBlock() because Block
// mode may have split the body.  Leaves the builder at retBb.
void emitTaskIdLoop(llvm::IRBuilder<> &builder, KernelCodegen &cg, const LaunchArgs &args,
                    const FunctionSkeleton &skel) {
  builder.CreateBr(skel.loopBb);
  builder.SetInsertPoint(skel.loopBb);
  auto *taskId = builder.CreatePHI(builder.getInt64Ty(), 2);
  taskId->addIncoming(args.ctrBegin, skel.entryBb);
  builder.CreateCondBr(builder.CreateICmpSLT(taskId, args.ctrEnd), skel.loopBodyBb, skel.retBb);

  builder.SetInsertPoint(skel.loopBodyBb);
  auto *ptrSvBegin = cg.emitSvBasePtr(args.p_sv, taskId);
  cg.emitLoopBody(ptrSvBegin);

  auto *taskIdNext = builder.CreateAdd(taskId, builder.getInt64(1));
  auto *loopTailBb = builder.GetInsertBlock();
  taskId->addIncoming(taskIdNext, loopTailBb);
  builder.CreateBr(skel.loopBb);

  builder.SetInsertPoint(skel.retBb);
  builder.CreateRetVoid();
}

} // namespace

llvm::Expected<llvm::Function *>
cast::cpu::generateKernelIr(const cast::cpu::KernelGenSpec &spec, const cast::Complex64 *matrix,
                            size_t matrixLen, const uint32_t *qubits, size_t nQubits,
                            llvm::StringRef funcName, llvm::Module &module) {
  if (auto err = validateKernelGenInputs(spec, matrix, matrixLen, qubits, nQubits))
    return std::move(err);

  const unsigned s = cast::cpu::getSimdS(spec.simdWidth, spec.precision);
  assert(s > 0 && s <= 4);
  const MatrixView matView{matrix, static_cast<uint32_t>(1u << nQubits)};

  const BitLayout layout = computeBitLayout(qubits, nQubits, s);
  const unsigned simdWidthBytes = static_cast<unsigned>(spec.simdWidth) / 8u;
  const ShuffleMasks smasks = computeShuffleMasks(layout, layout.S(), layout.s(), layout.vecSize());

  llvm::IRBuilder<> builder(module.getContext());
  builder.setFastMathFlags(llvm::FastMathFlags::getFast());
  auto *scalarTy =
      (spec.precision == cast::Precision::F32) ? builder.getFloatTy() : builder.getDoubleTy();
  const TypeBundle types{scalarTy, llvm::VectorType::get(scalarTy, layout.vecSize(), false)};

  const FunctionSkeleton skel = createFunctionSkeleton(module, builder, funcName);
  builder.SetInsertPoint(skel.entryBb);
  const LaunchArgs args = unpackLaunchArgs(builder, skel.func, skel.launchTy);
  const auto matData = buildMatrixData(builder, spec, matView, args.p_mat, scalarTy, s);

  // Default 32 (matches AVX-512, NEON, SVE).  Override with CAST_CPU_VEC_REGS
  // for A/B benchmarking (e.g. 9999 → force Straight on all gates).
  // Note: `static` — read once at first call, cached for the process lifetime.
  static const unsigned vecRegs = [] {
    const char *s = std::getenv("CAST_CPU_VEC_REGS");
    if (s && s[0] != '\0') {
      char *end = nullptr;
      const long v = std::strtol(s, &end, 10);
      if (end && *end == '\0' && v >= 2)
        return static_cast<unsigned>(v);
    }
    return 32u;
  }();

  // Default Mega.  The optimal mode depends on ISA-specific shuffle vs
  // scalar-gather costs; benchmarking is needed per target to determine which
  // is better.  See docs/cpu_kernel_loadmode.md for background.
  static const auto loadMode = [] {
    const char *s = std::getenv("CAST_CPU_LOADMODE");
    if (s && std::strcmp(s, "tiled") == 0)
      return cast::cpu::LoadMode::Tiled;
    return cast::cpu::LoadMode::Mega;
  }();

  KernelCodegen cg(builder, layout, simdWidthBytes, smasks, matData, types, vecRegs, loadMode,
                   *skel.entryBb);

  emitTaskIdLoop(builder, cg, args, skel);
  return skel.func;
}
