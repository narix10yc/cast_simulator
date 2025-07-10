#ifndef CAST_CPU_CPUOPTIMIZERBUILDER_H
#define CAST_CPU_CPUOPTIMIZERBUILDER_H

#include "cast/Core/Optimize.h"
#include "cast/CPU/CPUFusionConfig.h"
#include "utils/MaybeError.h"

namespace cast {

class CPUOptimizerBuilder {
  // CPUOptimizerBuilder is likely to be used in main(). We want to keep the 
  // stack clean
  struct Data {
    CPUFusionConfig fusionConfig;

    bool enableCanonicalization = true;
    bool enableFusion = true;
  };
  std::unique_ptr<Data> data;
public:
  CPUOptimizerBuilder() : data(std::make_unique<Data>()) {}

  void setNThreads(int nThreads) {
    data->fusionConfig.nThreads = nThreads;
  }

  void setPrecision(Precision precision) {
    data->fusionConfig.precision = precision;
  }

  void setZeroTol(double zeroTol) {
    data->fusionConfig.zeroTol = zeroTol;
  }

  void setSwapTol(double swapTol) {
    data->fusionConfig.swapTol = swapTol;
  }

  void disableCanonicalization() {
    data->enableCanonicalization = false;
  }

  void disableFusion() {
    data->enableFusion = false;
  }

  void enableCanonicalization() {
    data->enableCanonicalization = true;
  }

  void enableFusion() {
    data->enableFusion = true;
  }

  MaybeError<Optimizer> build();
  
}; // class CPUOptimizerBuilder

} // end namespace cast

#endif // CAST_CPU_CPUOPTIMIZERBUILDER_H