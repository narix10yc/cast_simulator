#include "matrix_data.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>

namespace cast::cpu {

std::vector<IRMatData> buildMatrixData(llvm::IRBuilder<> &builder, const KernelGenSpec &spec,
                                       const MatrixView &matrix, llvm::Value *pMatArg,
                                       llvm::Type *scalarTy, unsigned simdS) {
  const unsigned kk = matrix.edgeSize * matrix.edgeSize;
  const auto ec = llvm::ElementCount::getFixed(1u << simdS);

  std::vector<IRMatData> out(kk);
  for (unsigned i = 0; i < kk; ++i) {
    if (spec.mode == MatrixLoadMode::ImmValue) {
      out[i].reVec =
          llvm::ConstantVector::getSplat(ec, llvm::ConstantFP::get(scalarTy, matrix.re(i)));
      out[i].imVec =
          llvm::ConstantVector::getSplat(ec, llvm::ConstantFP::get(scalarTy, matrix.im(i)));
    } else {
      auto *rePtr = builder.CreateConstGEP1_32(scalarTy, pMatArg, 2 * i);
      out[i].reVec = builder.CreateVectorSplat(ec, builder.CreateLoad(scalarTy, rePtr));
      auto *imPtr = builder.CreateConstGEP1_32(scalarTy, pMatArg, 2 * i + 1);
      out[i].imVec = builder.CreateVectorSplat(ec, builder.CreateLoad(scalarTy, imPtr));
    }
  }
  return out;
}

} // namespace cast::cpu
