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
  std::string func_name;
  std::unique_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::Module> module;
  std::vector<cast::Complex64> matrix;
  /// Populated by optimize_kernel_ir; empty until then.
  std::string ir;
  bool optimized = false;
  /// Set from the `capture_ir` field of the kernel generation request.
  /// When true, jit_compile_kernel copies the optimized IR text into the
  /// returned CompiledKernelRecord.
  bool capture_ir = false;
  /// Set from the `capture_asm` field of the kernel generation request.
  /// When false, jit_compile_kernel skips assembly emission.
  bool capture_asm = false;
};

llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>> jit_create(unsigned n_compile_threads);

/// Runs the O1 pass pipeline on the plain Module inside `generated`.
/// Caches the resulting IR text in `generated.ir`.
/// Idempotent: a second call is a no-op.
/// Must be called before jit_compile_kernel, which moves the Module out.
llvm::Error optimize_kernel_ir(GeneratedKernel &generated);

/// Compiles `generated` into the JIT and returns per-kernel data.
llvm::Expected<CompiledKernelRecord> jit_compile_kernel(llvm::orc::LLJIT &jit,
                                                        GeneratedKernel &generated);

} // namespace cast::cpu

#endif // CAST_SIMULATOR_SRC_CPP_CPU_JIT_HPP
