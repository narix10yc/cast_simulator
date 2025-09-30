#ifndef CAST_CPU_CPUOPTIMIZER_H
#define CAST_CPU_CPUOPTIMIZER_H

#include "cast/CPU/CPUFusionConfig.h"
#include "cast/Core/Optimizer.h"

#include "llvm/Support/Casting.h"

namespace cast {

class CPUOptimizer : public Optimizer<CPUOptimizer> {

public:
  CPUOptimizer() = default;

  llvm::Error loadCPUCostModel(const std::string& filename,
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

  void displayInfo(utils::InfoLogger logger) const {
    logger.put("CPUOptimizer");
    if (fusionConfig_) {
      logger.put("Fusion Config");
      auto l = logger.indent();
      fusionConfig_->displayInfo(l);
    } else {
      logger.put("Fusion Config", "None");
    }
    logger.put("Canonicalization",
               enableCanonicalization_ ? "Enabled" : "Disabled");
    logger.put("Fusion", enableFusion_ ? "Enabled" : "Disabled");
    logger.put("CFO", enableCFO_ ? "Enabled" : "Disabled");
  }

}; // class CPUOptimizer

} // end namespace cast

#endif // CAST_CPU_CPUOPTIMIZER_H