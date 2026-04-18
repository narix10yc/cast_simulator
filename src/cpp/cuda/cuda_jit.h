#ifndef CAST_SIMULATOR_SRC_CPP_CUDA_JIT_H
#define CAST_SIMULATOR_SRC_CPP_CUDA_JIT_H

#include "../include/ffi_cuda.h"

#include <cstdint>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Target/TargetMachine.h>

#include <memory>
#include <string>
#include <vector>

/// IR generated for one kernel, not yet compiled.
struct CastCudaGeneratedKernel {
  cast_cuda_kernel_gen_spec_t spec{};
  uint32_t n_gate_qubits = 0;
  std::string func_name;
  std::unique_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::Module> module;
  std::unique_ptr<llvm::TargetMachine> tm; ///< created once, reused for PTX emission
  std::string ir;                          ///< cached after cast_cuda_optimize_kernel_ir
  bool optimized = false;
};

/// Result of compiling one kernel: PTX text.
struct CastCudaCompiledKernel {
  uint32_t n_gate_qubits = 0;
  cast_precision_t precision = CAST_PRECISION_F64;
  std::string func_name;
  std::string ptx;
};

/// Runs O1 with the NVPTX target machine, caches the IR text, sets
/// triple + data layout on the module.  Idempotent: a second call is a no-op.
llvm::Error cast_cuda_optimize_kernel_ir(CastCudaGeneratedKernel &generated);

/// Full pipeline: optimize → PTX (via NVPTX addPassesToEmitFile).
/// On success the compiled data lives in `out`.
llvm::Error cast_cuda_compile_kernel(CastCudaGeneratedKernel &generated,
                                     CastCudaCompiledKernel &out);

#endif // CAST_SIMULATOR_SRC_CPP_CUDA_JIT_H
