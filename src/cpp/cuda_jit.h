#ifndef CAST_SIMULATOR_SRC_CPP_CUDA_JIT_H
#define CAST_SIMULATOR_SRC_CPP_CUDA_JIT_H

#include "cuda.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/// IR generated for one kernel, not yet compiled.
/// Owned by cast_cuda_kernel_generator_t; moved out during finish().
struct CastCudaGeneratedKernel {
  cast_cuda_kernel_gen_spec_t spec{};
  cast_cuda_kernel_id_t       kernel_id = 0;
  uint32_t                    n_gate_qubits = 0;
  std::string                 func_name{};
  std::unique_ptr<llvm::LLVMContext> context{};
  std::unique_ptr<llvm::Module>      module{};
  std::string ir{};      ///< cached after cast_cuda_optimize_kernel_ir
  bool        optimized = false;
};

/// Result of compiling one kernel: PTX text + cubin bytes.
struct CastCudaCompiledKernel {
  cast_cuda_kernel_id_t  kernel_id = 0;
  uint32_t               n_gate_qubits = 0;
  cast_cuda_precision_t  precision = CAST_CUDA_PRECISION_F64;
  std::string            func_name{};
  std::string            ptx{};
  std::vector<uint8_t>   cubin{};
};

/// Compilation session: keyed by kernel_id for O(1) lookup.
/// Defined here (not in cuda.cpp) so cuda_exec.cpp can access the kernels map.
struct cast_cuda_compilation_session_t {
  std::unordered_map<cast_cuda_kernel_id_t, CastCudaCompiledKernel> kernels{};
};

/// Runs O1 with the NVPTX target machine, caches the IR text, sets
/// triple + data layout on the module.  Idempotent: a second call is a no-op.
llvm::Error cast_cuda_optimize_kernel_ir(CastCudaGeneratedKernel &generated);

/// Full pipeline: optimize → PTX (via NVPTX addPassesToEmitFile) →
/// cubin (via nvJitLink).  On success the compiled data lives in `out`.
llvm::Error cast_cuda_compile_kernel(CastCudaGeneratedKernel &generated,
                                     CastCudaCompiledKernel  &out);

#endif // CAST_SIMULATOR_SRC_CPP_CUDA_JIT_H
