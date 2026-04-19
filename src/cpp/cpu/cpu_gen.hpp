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

llvm::Expected<llvm::Function *> generateKernelIr(const KernelGenSpec &spec,
                                                  const cast::Complex64 *matrix, size_t matrixLen,
                                                  const uint32_t *qubits, size_t nQubits,
                                                  llvm::StringRef funcName, llvm::Module &module);

} // namespace cast::cpu

#endif // CAST_SIMULATOR_SRC_CPP_CPU_GEN_HPP
