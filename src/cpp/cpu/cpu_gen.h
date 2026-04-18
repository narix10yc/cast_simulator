#ifndef CAST_SIMULATOR_SRC_CPP_CPU_GEN_H
#define CAST_SIMULATOR_SRC_CPP_CPU_GEN_H

#include "../include/ffi_cpu.h"

#include <llvm/Support/Error.h>

namespace llvm {
class Function;
class Module;
class StringRef;
} // namespace llvm

llvm::Expected<llvm::Function *>
cast_cpu_generate_kernel_ir(const cast_cpu_kernel_gen_spec_t &spec, const cast_complex64_t *matrix,
                            size_t matrix_len, const uint32_t *qubits, size_t n_qubits,
                            llvm::StringRef func_name, llvm::Module &module);

#endif // CAST_SIMULATOR_SRC_CPP_CPU_GEN_H
