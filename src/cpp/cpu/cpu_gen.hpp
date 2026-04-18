#ifndef CAST_SIMULATOR_SRC_CPP_CPU_GEN_HPP
#define CAST_SIMULATOR_SRC_CPP_CPU_GEN_HPP

#include "../internal/types.hpp"

#include <llvm/Support/Error.h>

#include <cstddef>
#include <cstdint>

namespace llvm {
class Function;
class Module;
class StringRef;
} // namespace llvm

namespace cast::cpu {

llvm::Expected<llvm::Function *> generate_kernel_ir(const KernelGenSpec &spec,
                                                    const cast::Complex64 *matrix,
                                                    size_t matrix_len, const uint32_t *qubits,
                                                    size_t n_qubits, llvm::StringRef func_name,
                                                    llvm::Module &module);

} // namespace cast::cpu

#endif // CAST_SIMULATOR_SRC_CPP_CPU_GEN_HPP
