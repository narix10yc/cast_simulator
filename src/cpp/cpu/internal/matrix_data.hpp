#ifndef CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_MATRIX_DATA_HPP
#define CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_MATRIX_DATA_HPP

#include "../../internal/types.hpp"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cast::cpu {

// View into a flat row-major complex matrix.
struct MatrixView {
  const cast::Complex64 *data = nullptr;
  uint32_t edgeSize = 0;

  double re(size_t idx) const { return data[idx].re; }
  double im(size_t idx) const { return data[idx].im; }
};

// One re/im vector pair per matrix element.  Populated by
// `buildMatrixData`; consumed by Phase 2 matvec emission.
struct IRMatData {
  llvm::Value *reVec = nullptr;
  llvm::Value *imVec = nullptr;
};

// ImmValue → splatted ConstantFP per matrix entry; StackLoad → runtime
// load+splat from the caller's matrix buffer.
std::vector<IRMatData> buildMatrixData(llvm::IRBuilder<> &builder, const KernelGenSpec &spec,
                                       const MatrixView &matrix, llvm::Value *pMatArg,
                                       llvm::Type *scalarTy, unsigned simdS);

} // namespace cast::cpu

#endif // CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_MATRIX_DATA_HPP
