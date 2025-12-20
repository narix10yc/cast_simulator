#ifndef CAST_CPU_CPUFUSIONCONFIG_H
#define CAST_CPU_CPUFUSIONCONFIG_H

#include "cast/CPU/CPUCostModel.h"
#include "cast/Core/FusionConfig.h"

#include <llvm/Support/Casting.h>

namespace cast {

// CPUFusionConfig is strongly associated with CPUCostModel.
class CPUFusionConfig : public FusionConfig {
public:
  CPUFusionConfig(std::unique_ptr<CPUCostModel> cpuCM) : FusionConfig(FC_CPU) {
    assert(cpuCM != nullptr);
    this->costModel_ = std::move(cpuCM);
  }

  void displayInfo(utils::InfoLogger logger) const override;

  static bool classof(const FusionConfig* config) {
    return config->getKind() == FC_CPU;
  }

}; // class CPUFusionConfig

} // end namespace cast

#endif // CAST_CPU_CPUFUSIONCONFIG_H