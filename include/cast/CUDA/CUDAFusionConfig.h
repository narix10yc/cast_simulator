#ifndef CAST_CUDA_CUDAFUSIONCONFIG_H
#define CAST_CUDA_CUDAFUSIONCONFIG_H

#include "cast/Core/FusionConfig.h"
#include "llvm/Support/Casting.h"
#include <memory>

namespace cast {

class CUDAFusionConfig : public FusionConfig {
public:
  CUDAFusionConfig() : FusionConfig(FC_CUDA) {}

  static bool classof(const FusionConfig* base) {
    return base->getKind() == FC_CUDA;
  }
};

} // namespace cast

#endif // CAST_CUDA_CUDAFUSIONCONFIG_H
