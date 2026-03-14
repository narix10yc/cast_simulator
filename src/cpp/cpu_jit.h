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

using cast_cpu_kernel_entry_t = void(void*);

struct CastCpuGeneratedKernel {
  cast_cpu_kernel_metadata_t metadata{};
  std::string func_name{};
  std::unique_ptr<llvm::LLVMContext> context{};
  std::unique_ptr<llvm::Module> module{};
  std::vector<cast_cpu_complex64_t> matrix{};
};

struct CastCpuJittedKernel {
  cast_cpu_kernel_metadata_t metadata{};
  std::string func_name{};
  cast_cpu_kernel_entry_t* entry = nullptr;
  std::vector<cast_cpu_complex64_t> matrix{};
};

llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>>
cast_cpu_jit_create(unsigned n_compile_threads);

llvm::Error cast_cpu_jit_compile_kernel(llvm::orc::LLJIT& jit,
                                        CastCpuGeneratedKernel& generated,
                                        CastCpuJittedKernel& out);

llvm::Error cast_cpu_jit_apply_kernel(const CastCpuJittedKernel& kernel,
                                      void* sv,
                                      uint32_t n_qubits_sv,
                                      cast_cpu_precision_t sv_precision,
                                      cast_cpu_simd_width_t sv_simd_width,
                                      int n_threads);

#endif // CAST_SIMULATOR_SRC_CPP_CPU_JIT_H
