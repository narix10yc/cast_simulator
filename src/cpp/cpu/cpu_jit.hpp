#ifndef CAST_SIMULATOR_SRC_CPP_CPU_JIT_HPP
#define CAST_SIMULATOR_SRC_CPP_CPU_JIT_HPP

#include "../internal/types.hpp"

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>

#include <memory>
#include <string>
#include <vector>

namespace cast::cpu {

struct GeneratedKernel {
  KernelMetadata metadata{};
  std::string funcName;
  std::unique_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::Module> module;
  std::vector<cast::Complex64> matrix;
  /// Populated by optimizeKernelIr; empty until then.
  std::string ir;
  bool optimized = false;
  /// Set from the `captureIr` field of the kernel generation request.
  /// When true, jitCompileKernel copies the optimized IR text into the
  /// returned CompiledKernelRecord.
  bool captureIr = false;
  /// Set from the `captureAsm` field of the kernel generation request.
  /// When false, jitCompileKernel skips assembly emission.
  bool captureAsm = false;
};

llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>> createJit(unsigned nCompileThreads);

/// Runs the O1 pass pipeline on the plain Module inside `generated`.
/// Caches the resulting IR text in `generated.ir`.
/// Idempotent: a second call is a no-op.
/// Must be called before jitCompileKernel, which moves the Module out.
llvm::Error optimizeKernelIr(GeneratedKernel &generated);

/// Compiles `generated` into the JIT and returns per-kernel data.
llvm::Expected<CompiledKernelRecord> jitCompileKernel(llvm::orc::LLJIT &jit,
                                                      GeneratedKernel &generated);

} // namespace cast::cpu

#endif // CAST_SIMULATOR_SRC_CPP_CPU_JIT_HPP
