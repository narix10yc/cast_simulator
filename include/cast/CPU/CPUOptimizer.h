#ifndef CAST_CPU_CPUOPTIMIZER_H
#define CAST_CPU_CPUOPTIMIZER_H

#include "cast/Core/Optimizer.h"

namespace cast {

class CPUOptimizer : public Optimizer<CPUOptimizer> {

public:
  CPUOptimizer() = default;

  llvm::Error loadCPUCostModel(const std::string& cacheFilename,
                               int queryNThreads,
                               Precision queryPrecision);

  void displayInfo(utils::InfoLogger logger) const;

}; // class CPUOptimizer

} // end namespace cast

#endif // CAST_CPU_CPUOPTIMIZER_H