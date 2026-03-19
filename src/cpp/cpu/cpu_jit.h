#ifndef CAST_SIMULATOR_SRC_CPP_CPU_JIT_H
#define CAST_SIMULATOR_SRC_CPP_CPU_JIT_H

#include "cpu.h"

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>

#include <memory>
#include <string>
#include <vector>

struct CastCpuGeneratedKernel {
  cast_cpu_kernel_metadata_t metadata{};
  std::string func_name{};
  std::unique_ptr<llvm::LLVMContext> context{};
  std::unique_ptr<llvm::Module> module{};
  std::vector<cast_cpu_complex64_t> matrix{};
  /// Populated by cast_cpu_optimize_kernel_ir; empty until then.
  std::string ir{};
  bool optimized = false;
  /// Set by cast_cpu_kernel_generator_request_asm before finish().
  /// When false, cast_cpu_jit_compile_kernel skips assembly emission.
  bool capture_asm = false;
};

llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>> cast_cpu_jit_create(unsigned n_compile_threads);

/// Runs the O1 pass pipeline on the plain Module inside `generated`.
/// Caches the resulting IR text in `generated.ir`.
/// Idempotent: a second call is a no-op.
/// Must be called before cast_cpu_jit_compile_kernel, which moves the Module out.
llvm::Error cast_cpu_optimize_kernel_ir(CastCpuGeneratedKernel &generated);

/// Compiles `generated` into the JIT and writes per-kernel data into `out`.
/// Heap-allocates out.matrix (for StackLoad mode) and out.asm_text (if
/// capture_asm is set); the caller is responsible for freeing them (e.g. via
/// cast_cpu_jit_kernel_records_free).
llvm::Error cast_cpu_jit_compile_kernel(llvm::orc::LLJIT &jit,
                                        CastCpuGeneratedKernel &generated,
                                        cast_cpu_jit_kernel_record_t &out);

#endif // CAST_SIMULATOR_SRC_CPP_CPU_JIT_H
