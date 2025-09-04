#ifndef CAST_CUDA_CUDAFUSIONCONFIG_H
#define CAST_CUDA_CUDAFUSIONCONFIG_H

#include "cast/Core/FusionConfig.h"

namespace cast {

class CUDAFusionConfig : public FusionConfig {
public:
  CUDAFusionConfig() : FusionConfig(FC_CUDA) {}

  std::ostream& displayInfo(std::ostream& os, int verbose) const override {
    return os << "CUDA Fusion Config\n";
  }

  static bool classof(const FusionConfig* base) {
    return base->getKind() == FC_CUDA;
  }
};

} // namespace cast

#endif // CAST_CUDA_CUDAFUSIONCONFIG_H
