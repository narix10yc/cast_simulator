#ifndef CAST_CUDA_CUDAOPTIMIZER_H
#define CAST_CUDA_CUDAOPTIMIZER_H

#include "cast/CUDA/CUDACostModel.h"
#include "cast/CUDA/CUDAFusionConfig.h"
#include "cast/Core/Optimizer.h"

namespace cast {

class CUDAOptimizer : public Optimizer<CUDAOptimizer> {
public:
  CUDAOptimizer() = default;

  llvm::Error loadCUDACostModelFromFile(const std::string& filename,
                                        Precision queryPrecision) {
    std::ifstream ifile(filename);
    auto expectedCache = CUDAPerformanceCache::LoadFrom(ifile);
    if (!expectedCache) {
      return llvm::joinErrors(
          llvm::createStringError(
              "Failed to load CUDA performance cache from " + filename),
          expectedCache.takeError());
    }
    auto cm = std::make_unique<CUDACostModel>(*expectedCache);
    cm->setQueryPrecision(queryPrecision);
    auto fc = std::make_unique<CUDAFusionConfig>();
    fc->setCostModel(std::move(cm));
    this->setFusionConfig(std::move(fc));
    return llvm::Error::success();
  }
};

} // namespace cast

#endif // CAST_CUDA_CUDAOPTIMIZER_H