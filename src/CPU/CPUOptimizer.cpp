#include "cast/CPU/CPUOptimizer.h"
#include "cast/CPU/CPUCostModel.h"
#include "cast/CPU/CPUFusionConfig.h"
#include "llvm/Support/Error.h"

using namespace cast;

llvm::Error CPUOptimizer::loadCPUCostModel(const std::string& cacheFilename,
                                           int queryNThreads,
                                           Precision queryPrecision) {
  auto cache = CPUPerformanceCache::LoadFromFile(cacheFilename);
  if (cache == nullptr) {
    return llvm::createStringError("Unable to load performance cache from " +
                                   cacheFilename);
  }
  auto cm = std::make_unique<CPUCostModel>(queryNThreads, queryPrecision);
  if (auto e = cm->loadCache(*cache)) {
    return llvm::joinErrors(llvm::createStringError(__PRETTY_FUNCTION__),
                            std::move(e));
  }

  auto fc = std::make_unique<CPUFusionConfig>(std::move(cm));
  this->setFusionConfig(std::move(fc));

  return llvm::Error::success();
}

void CPUOptimizer::displayInfo(utils::InfoLogger logger) const {
  logger.put("CPUOptimizer");
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