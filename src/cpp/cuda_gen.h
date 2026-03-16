#ifndef CAST_SIMULATOR_SRC_CPP_CUDA_GEN_H
#define CAST_SIMULATOR_SRC_CPP_CUDA_GEN_H

#include "cuda.h"

#include <llvm/Support/Error.h>

namespace llvm {
class Function;
class Module;
class StringRef;
} // namespace llvm

/// Generates LLVM NVPTX IR for one CUDA gate kernel.
///
/// The produced function has signature:
///   void @func_name(ptr sv, ptr mat, i64 combos)
/// where each GPU thread processes a strided slice of the "combos"
/// (amplitude index combinations).  For ImmValue mode the matrix pointer
/// is unused at runtime — all matrix elements are baked as constants.
llvm::Expected<llvm::Function *>
cast_cuda_generate_kernel_ir(const cast_cuda_kernel_gen_spec_t &spec,
                             const cast_cuda_complex64_t *matrix,
                             size_t matrix_len,
                             const uint32_t *qubits, size_t n_qubits,
                             llvm::StringRef func_name,
                             llvm::Module &module);

#endif // CAST_SIMULATOR_SRC_CPP_CUDA_GEN_H
