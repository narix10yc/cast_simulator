#ifndef CAST_CPU_CPUOPTIMIZER_H
#define CAST_CPU_CPUOPTIMIZER_H

#include "cast/CPU/CPUFusionConfig.h"
#include "cast/Core/Optimizer.h"
#include "utils/MaybeError.h"

#include "llvm/Support/Casting.h"

namespace cast {

class CPUOptimizer : public Optimizer {
  // Allowed configs are: SizeOnlyFusionConfig, CPUFusionConfig
  // Default is SizeOnlyFusionConfig with size 3.
  std::unique_ptr<FusionConfig> fusionConfig;

  bool enableCanonicalization_ = true;
  bool enableFusion_ = true;
  bool enableCFO_ = true;

public:
  CPUOptimizer() : fusionConfig(std::make_unique<SizeOnlyFusionConfig>(3)) {}

  CPUOptimizer& setSizeOnlyFusionConfig(int size) {
    fusionConfig = std::make_unique<SizeOnlyFusionConfig>(size);
    return *this;
  }

  CPUOptimizer& setCPUFusionConfig(std::unique_ptr<CPUFusionConfig> config) {
    fusionConfig = std::move(config);
    return *this;
  }

  // Only meaningful for CPUFusionConfig
  CPUOptimizer& setNThreads(int nThreads) {
    if (auto* cpuFusionConfig =
            llvm::dyn_cast<CPUFusionConfig>(fusionConfig.get())) {
      cpuFusionConfig->setNThreads(nThreads);
    }
    return *this;
  }

  // Only meaningful for CPUFusionConfig
  CPUOptimizer& setPrecision(Precision precision) {
    if (auto* cpuFusionConfig =
            llvm::dyn_cast<CPUFusionConfig>(fusionConfig.get())) {
      cpuFusionConfig->setPrecision(precision);
    }
    return *this;
  }

  CPUOptimizer& setZeroTol(double zeroTol) {
    if (fusionConfig)
      fusionConfig->zeroTol = zeroTol;
    return *this;
  }

  CPUOptimizer& setSwapTol(double swapTol) {
    fusionConfig->swapTol = swapTol;
    return *this;
  }

  CPUOptimizer& disableCanonicalization() {
    enableCanonicalization_ = false;
    return *this;
  }

  CPUOptimizer& disableFusion() {
    enableFusion_ = false;
    return *this;
  }

  CPUOptimizer& disableCFO() {
    enableCFO_ = false;
    return *this;
  }

  CPUOptimizer& enableCanonicalization() {
    enableCanonicalization_ = true;
    return *this;
  }

  CPUOptimizer& enableFusion() {
    enableFusion_ = true;
    return *this;
  }

  CPUOptimizer& enableCFO() {
    enableCFO_ = true;
    return *this;
  }

  void run(ir::CircuitNode& circuit, utils::Logger logger) const override;

  void run(ir::CircuitGraphNode& graph, utils::Logger logger) const override;

}; // class CPUOptimizer

} // end namespace cast

#endif // CAST_CPU_CPUOPTIMIZER_H