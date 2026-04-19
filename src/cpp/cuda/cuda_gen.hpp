#ifndef CAST_SIMULATOR_SRC_CPP_CUDA_GEN_HPP
#define CAST_SIMULATOR_SRC_CPP_CUDA_GEN_HPP

#include "../internal/types.hpp"

#include <llvm/Support/Error.h>

#include <cstddef>
#include <cstdint>

namespace llvm {
class Function;
class Module;
class StringRef;
} // namespace llvm

namespace cast::cuda {

/// Generates LLVM NVPTX IR for one CUDA gate kernel.
///
/// The produced function has signature:
///   void @funcName(ptr sv, ptr mat, i64 combos)
/// where each GPU thread processes a strided slice of the "combos"
/// (amplitude index combinations).  For ImmValue mode the matrix pointer
/// is unused at runtime — all matrix elements are baked as constants.
llvm::Expected<llvm::Function *> generateKernelIr(const KernelGenSpec &spec,
                                                  const cast::Complex64 *matrix, size_t matrixLen,
                                                  const uint32_t *qubits, size_t nQubits,
                                                  llvm::StringRef funcName, llvm::Module &module);

} // namespace cast::cuda

#endif // CAST_SIMULATOR_SRC_CPP_CUDA_GEN_HPP
