#ifndef CAST_CUDA_CUDAFUSIONCONFIG_H
#define CAST_CUDA_CUDAFUSIONCONFIG_H

#include "cast/CUDA/CUDACostModel.h"
#include "cast/Core/FusionConfig.h"

#include "llvm/Support/Casting.h"

namespace cast {

class CUDAFusionConfig : public FusionConfig {
public:
  CUDAFusionConfig(std::unique_ptr<CUDACostModel> cudaCostModel,
                   Precision precision = Precision::Unknown)
      : FusionConfig(FC_CUDA) {
    this->costModel = std::move(cudaCostModel);
    setPrecision(precision);
  }

  void setPrecision(Precision precision) {
    if (auto *cudaCM = llvm::dyn_cast<CUDACostModel>(costModel.get()))
      cudaCM->setQueryPrecision(precision);
  }

  std::ostream &displayInfo(std::ostream &os, int verbose = 1) const override;

  /// RTTI hook so `llvm::dyn_cast<CUDAFusionConfig>(config)` works.
  static bool classof(const FusionConfig *config) {
    return config->getKind() == FC_CUDA;
  }
};

} // namespace cast

#endif // CAST_CUDA_CUDAFUSIONCONFIG_H