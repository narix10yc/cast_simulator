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
///   void @func_name(ptr sv, ptr mat, i64 combos)
/// where each GPU thread processes a strided slice of the "combos"
/// (amplitude index combinations).  For ImmValue mode the matrix pointer
/// is unused at runtime — all matrix elements are baked as constants.
llvm::Expected<llvm::Function *> generate_kernel_ir(const KernelGenSpec &spec,
                                                    const cast::Complex64 *matrix,
                                                    size_t matrix_len, const uint32_t *qubits,
                                                    size_t n_qubits, llvm::StringRef func_name,
                                                    llvm::Module &module);

} // namespace cast::cuda

#endif // CAST_SIMULATOR_SRC_CPP_CUDA_GEN_HPP
