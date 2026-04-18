#ifndef CAST_SIMULATOR_SRC_CPP_CUDA_JIT_HPP
#define CAST_SIMULATOR_SRC_CPP_CUDA_JIT_HPP

#include "../internal/types.hpp"

#include <cstdint>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Target/TargetMachine.h>

#include <memory>
#include <string>
#include <vector>

namespace cast::cuda {

/// IR generated for one kernel, not yet compiled.
struct GeneratedKernel {
  KernelGenSpec spec{};
  uint32_t n_gate_qubits = 0;
  std::string func_name;
  std::unique_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::Module> module;
  std::unique_ptr<llvm::TargetMachine> tm; ///< created once, reused for PTX emission
  std::string ir;                          ///< cached after optimize_kernel_ir
  bool optimized = false;
};

/// Result of compiling one kernel: PTX text.
struct CompiledKernel {
  uint32_t n_gate_qubits = 0;
  cast::Precision precision = cast::Precision::F64;
  std::string func_name;
  std::string ptx;
};

/// Runs O1 with the NVPTX target machine, caches the IR text, sets
/// triple + data layout on the module.  Idempotent: a second call is a no-op.
llvm::Error optimize_kernel_ir(GeneratedKernel &generated);

/// Full pipeline: optimize → PTX (via NVPTX addPassesToEmitFile).
/// On success the compiled data lives in `out`.
llvm::Error compile_kernel(GeneratedKernel &generated, CompiledKernel &out);

} // namespace cast::cuda

#endif // CAST_SIMULATOR_SRC_CPP_CUDA_JIT_HPP
