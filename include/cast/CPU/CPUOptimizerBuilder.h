#ifndef CAST_CPU_CPUOPTIMIZERBUILDER_H
#define CAST_CPU_CPUOPTIMIZERBUILDER_H

#include "cast/Core/Optimize.h"
#include "cast/CPU/CPUFusionConfig.h"
#include "utils/MaybeError.h"

#include "llvm/Support/Casting.h"

namespace cast {

class CPUOptimizerBuilder {
  // CPUOptimizerBuilder is likely to be used in main(). We want to keep the 
  // stack clean
  struct Data {
    std::unique_ptr<FusionConfig> fusionConfig;

    bool enableCanonicalization = true;
    bool enableFusion = true;
    bool enableCFO = true;
    int verbose = 0;
  };
  std::unique_ptr<Data> data;

public:
  CPUOptimizerBuilder() : data(std::make_unique<Data>()) {
    data->fusionConfig = std::make_unique<CPUFusionConfig>();
  }

  CPUOptimizerBuilder& setSizeOnlyFusion(int size) {
    data->fusionConfig = std::make_unique<SizeOnlyFusionConfig>(size);
    return *this;
  }

  // Only meaningful for CPUFusionConfig
  CPUOptimizerBuilder& setNThreads(int nThreads) {
    if (auto* cpuFusionConfig = 
        llvm::dyn_cast<CPUFusionConfig>(data->fusionConfig.get())) {
      cpuFusionConfig->nThreads = nThreads;
    }
    return *this;
  }

  // Only meaningful for CPUFusionConfig
  CPUOptimizerBuilder& setPrecision(Precision precision) {
    if (auto* cpuFusionConfig = 
        llvm::dyn_cast<CPUFusionConfig>(data->fusionConfig.get())) {
      cpuFusionConfig->precision = precision;
    }
    return *this;
  }

  CPUOptimizerBuilder& setZeroTol(double zeroTol) {
    if (data->fusionConfig)
      data->fusionConfig->zeroTol = zeroTol;
    return *this;
  }

  CPUOptimizerBuilder& setSwapTol(double swapTol) {
    data->fusionConfig->swapTol = swapTol;
    return *this;
  }

  CPUOptimizerBuilder& disableCanonicalization() {
    data->enableCanonicalization = false;
    return *this;
  }

  CPUOptimizerBuilder& disableFusion() {
    data->enableFusion = false;
    return *this;
  }

  CPUOptimizerBuilder& disableCFO() {
    data->enableCFO = false;
    return *this;
  }

  CPUOptimizerBuilder& enableCanonicalization() {
    data->enableCanonicalization = true;
    return *this;
  }

  CPUOptimizerBuilder& enableFusion() {
    data->enableFusion = true;
    return *this;
  }

  CPUOptimizerBuilder& enableCFO() {
    data->enableCFO = true;
    return *this;
  }

  CPUOptimizerBuilder& setVerbose(int verbose) {
    data->verbose = verbose;
    return *this;
  }

  MaybeError<Optimizer> build();
  
}; // class CPUOptimizerBuilder

} // end namespace cast

#endif // CAST_CPU_CPUOPTIMIZERBUILDER_H