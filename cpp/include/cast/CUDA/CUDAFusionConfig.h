#ifndef CAST_CUDA_CUDAFUSIONCONFIG_H
#define CAST_CUDA_CUDAFUSIONCONFIG_H

#include "cast/Core/FusionConfig.h"

namespace cast {

class CUDAFusionConfig : public FusionConfig {
public:
  CUDAFusionConfig() : FusionConfig(FC_CUDA) {}

  void displayInfo(utils::InfoLogger logger) const override {
    assert(false && "Not implemented yet");
  }

  static bool classof(const FusionConfig* base) {
    return base->getKind() == FC_CUDA;
  }
};

} // namespace cast

#endif // CAST_CUDA_CUDAFUSIONCONFIG_H
