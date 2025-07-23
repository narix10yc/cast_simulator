#ifndef CAST_CPU_CPUFUSIONCONFIG_H
#define CAST_CPU_CPUFUSIONCONFIG_H

#include "cast/Core/FusionConfig.h"
#include "cast/CPU/CPUCostModel.h"

#include "llvm/Support/Casting.h"

namespace cast {

// Must provide a CPUCostModel in the constructor.
class CPUFusionConfig : public FusionConfig {
public:
  CPUFusionConfig(std::unique_ptr<CPUCostModel> cpuCostModel,
                  int nThreads = -1, // -1 means not set
                  Precision precision = Precision::Unknown)
    : FusionConfig(FC_CPU) {
    this->costModel = std::move(cpuCostModel);
    setNThreads(nThreads);
    setPrecision(precision);
  }

  void setNThreads(int nThreads) {
    if (auto* cpuCostModel = 
        llvm::dyn_cast<CPUCostModel>(costModel.get())) {
      cpuCostModel->setQueryNThreads(nThreads);
    }
  }

  void setPrecision(Precision precision) {
    if (auto* cpuCostModel = 
        llvm::dyn_cast<CPUCostModel>(costModel.get())) {
      cpuCostModel->setQueryPrecision(precision);
    }
  }

  std::ostream& displayInfo(std::ostream& os, int verbose = 1) const override;

  static bool classof(const FusionConfig* config) {
    return config->getKind() == FC_CPU;
  }

}; // class CPUFusionConfig

} // end namespace cast

#endif // CAST_CPU_CPUFUSIONCONFIG_H