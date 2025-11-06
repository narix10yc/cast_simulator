#include "cast/CPU/CPUOptimizer.h"
#include "cast/CPU/CPUFusionConfig.h"

using namespace cast;

llvm::Error CPUOptimizer::loadCPUCostModel(const std::string& filename,
                                           int queryNThreads,
                                           Precision queryPrecision) {
  auto fc = CPUFusionConfig::LoadFromFile(filename);
  if (fc == nullptr) {
    return llvm::createStringError("Unable to load cost model from " +
                                   filename);
  }
  // Fusion configs do not have queries. Cost models do.
  auto* cm = llvm::dyn_cast_or_null<CPUCostModel>(fc->getCostModel());
  if (cm == nullptr) {
    return llvm::createStringError("No CPU Cost Model found");
  }
  cm->setQueryNThreads(queryNThreads);
  cm->setQueryPrecision(queryPrecision);
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