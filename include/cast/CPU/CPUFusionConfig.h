#ifndef CAST_CPU_CPUFUSIONCONFIG_H
#define CAST_CPU_CPUFUSIONCONFIG_H

#include "cast/Core/FusionConfig.h"
#include "cast/CPU/CPUCostModel.h"

namespace cast {

class CPUFusionConfig : public FusionConfig {
public:
  CPUFusionConfig() : FusionConfig(FC_CPU) {}

  // Set to null to accept all fusion candidates.
  std::unique_ptr<CPUCostModel> cpuCostModel = nullptr;

  int nThreads = 0;
  Precision precision = Precision::Unknown;

  std::ostream& displayInfo(std::ostream& os, int verbose = 1) const override;

  static bool classof(const FusionConfig* config) {
    return config->getKind() == FC_CPU;
  }

}; // class CPUFusionConfig

} // end namespace cast

#endif // CAST_CPU_CPUFUSIONCONFIG_H