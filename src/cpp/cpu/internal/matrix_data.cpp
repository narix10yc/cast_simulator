#include "matrix_data.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>

namespace cast_cpu_detail {

std::vector<IRMatData> build_matrix_data(llvm::IRBuilder<> &builder,
                                         const cast_cpu_kernel_gen_spec_t &spec,
                                         const MatrixView &matrix, llvm::Value *p_mat_arg,
                                         llvm::Type *scalar_ty, unsigned simd_s) {
  const unsigned kk = matrix.edge_size * matrix.edge_size;
  const auto ec = llvm::ElementCount::getFixed(1u << simd_s);

  std::vector<IRMatData> out(kk);
  for (unsigned i = 0; i < kk; ++i) {
    if (spec.mode == CAST_CPU_MATRIX_LOAD_IMM_VALUE) {
      out[i].re_vec =
          llvm::ConstantVector::getSplat(ec, llvm::ConstantFP::get(scalar_ty, matrix.re(i)));
      out[i].im_vec =
          llvm::ConstantVector::getSplat(ec, llvm::ConstantFP::get(scalar_ty, matrix.im(i)));
    } else {
      auto *re_ptr = builder.CreateConstGEP1_32(scalar_ty, p_mat_arg, 2 * i, "re.mat.ptr");
      out[i].re_vec = builder.CreateVectorSplat(ec, builder.CreateLoad(scalar_ty, re_ptr, "re.mat"),
                                                "re.mat.vec");
      auto *im_ptr = builder.CreateConstGEP1_32(scalar_ty, p_mat_arg, 2 * i + 1, "im.mat.ptr");
      out[i].im_vec = builder.CreateVectorSplat(ec, builder.CreateLoad(scalar_ty, im_ptr, "im.mat"),
                                                "im.mat.vec");
    }
  }
  return out;
}

} // namespace cast_cpu_detail
