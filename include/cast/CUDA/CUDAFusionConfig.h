#ifndef CAST_CUDA_CUDAFUSIONCONFIG_H
#define CAST_CUDA_CUDAFUSIONCONFIG_H

#include "cast/Core/FusionConfig.h"
#include "cast/CUDA/CUDACostModel.h"
#include "llvm/Support/Casting.h"
#include <memory>

namespace cast {

class CUDAFusionConfig : public FusionConfig {
public:
  CUDAFusionConfig(std::unique_ptr<CUDACostModel> cudaCostModel,
                   int blockSize = -1,                       // -1 means not set
                   Precision precision = Precision::Unknown) // Unknown means not set
      : FusionConfig(FC_CUDA) {
    this->costModel = std::move(cudaCostModel);
    setBlockSize(blockSize);
    setPrecision(precision);
  }

  void setBlockSize(int bs) {
    if (auto* cm = llvm::dyn_cast<CUDACostModel>(costModel.get())) {
      if (bs > 0) cm->setQueryBlockSize(bs);
    }
  }

  void setPrecision(Precision p) {
    if (auto* cm = llvm::dyn_cast<CUDACostModel>(costModel.get())) {
      if (p != Precision::Unknown) cm->setQueryPrecision(p);
    }
  }

  CUDACostModel* getCUDACostModel() const {
    return llvm::dyn_cast<CUDACostModel>(costModel.get());
  }

  std::ostream& displayInfo(std::ostream& os, int verbose = 1) const override {
    if (auto* cm = llvm::dyn_cast<CUDACostModel>(costModel.get()))
      return cm->displayInfo(os, verbose);
    return os << "CUDAFusionConfig (no CUDA cost model)\n";
  }

  static bool classof(const FusionConfig* cfg) {
    return cfg->getKind() == FC_CUDA;
  }
};

} // namespace cast

#endif // CAST_CUDA_CUDAFUSIONCONFIG_H
