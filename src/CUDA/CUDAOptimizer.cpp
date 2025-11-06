#include "cast/CUDA/CUDAOptimizer.h"
#include "cast/CUDA/CUDACostModel.h"
#include "cast/CUDA/CUDAFusionConfig.h"

using namespace cast;

llvm::Error
CUDAOptimizer::loadCUDACostModelFromFile(const std::string& filename,
                                         Precision queryPrecision) {
  std::ifstream ifile(filename);
  auto expectedCache = CUDAPerformanceCache::LoadFrom(ifile);
  if (!expectedCache) {
    return llvm::joinErrors(
        llvm::createStringError("Failed to load CUDA performance cache from " +
                                filename),
        expectedCache.takeError());
  }
  auto cm = std::make_unique<CUDACostModel>(*expectedCache);
  cm->setQueryPrecision(queryPrecision);
  auto fc = std::make_unique<CUDAFusionConfig>();
  fc->setCostModel(std::move(cm));
  this->setFusionConfig(std::move(fc));
  return llvm::Error::success();
}

void CUDAOptimizer::displayInfo(utils::InfoLogger logger) const {
  logger.put("CUDAOptimizer");

  if (fusionConfig_) {
    logger.put("Fusion Config");
    fusionConfig_->displayInfo(logger.indent());
  } else {
    logger.put("Fusion Config", "None");
  }
  logger.put("Canonicalization",
             enableCanonicalization_ ? "Enabled" : "Disabled");
  logger.put("Fusion", enableFusion_ ? "Enabled" : "Disabled");
  logger.put("CFO", enableCFO_ ? "Enabled" : "Disabled");
}
