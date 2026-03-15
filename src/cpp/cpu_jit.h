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

using cast_cpu_kernel_entry_t = void(void *);

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

struct CastCpuJittedKernel {
  cast_cpu_kernel_metadata_t metadata{};
  std::string func_name{};
  cast_cpu_kernel_entry_t *entry = nullptr;
  std::vector<cast_cpu_complex64_t> matrix{};
  /// Native assembly text emitted during cast_cpu_jit_compile_kernel, before
  /// the module was moved into the JIT pipeline.
  std::string asm_text{};
};

llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>> cast_cpu_jit_create(unsigned n_compile_threads);

/// Runs the O1 pass pipeline on the plain Module inside `generated`.
/// Caches the resulting IR text in `generated.ir`.
/// Idempotent: a second call is a no-op.
/// Must be called before cast_cpu_jit_compile_kernel, which moves the Module out.
llvm::Error cast_cpu_optimize_kernel_ir(CastCpuGeneratedKernel &generated);

llvm::Error cast_cpu_jit_compile_kernel(llvm::orc::LLJIT &jit, CastCpuGeneratedKernel &generated,
                                        CastCpuJittedKernel &out);

llvm::Error cast_cpu_jit_apply_kernel(const CastCpuJittedKernel &kernel, void *sv,
                                      uint32_t n_qubits_sv, cast_cpu_precision_t sv_precision,
                                      cast_cpu_simd_width_t sv_simd_width, int n_threads);

#endif // CAST_SIMULATOR_SRC_CPP_CPU_JIT_H
