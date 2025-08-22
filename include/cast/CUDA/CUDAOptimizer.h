#pragma once
#include "cast/Core/Optimizer.h"
#include "cast/CUDA/CUDAFusionConfig.h"

namespace cast {

class CUDAOptimizer : public Optimizer {
  std::unique_ptr<FusionConfig> fusionConfig;
  bool enableCanonicalization_ = true;
  bool enableFusion_ = true;
  bool enableCFO_ = true;

public:
  CUDAOptimizer() = default;

  CUDAOptimizer& setCUDAFusionConfig(std::unique_ptr<CUDAFusionConfig> cfg) {
    fusionConfig = std::move(cfg);
    return *this;
  }

  CUDAOptimizer& enableCanonicalization() { enableCanonicalization_ = true; return *this; }
  CUDAOptimizer& disableCanonicalization() { enableCanonicalization_ = false; return *this; }
  CUDAOptimizer& enableFusion() { enableFusion_ = true; return *this; }
  CUDAOptimizer& disableFusion() { enableFusion_ = false; return *this; }
  CUDAOptimizer& enableCFO() { enableCFO_ = true; return *this; }
  CUDAOptimizer& disableCFO() { enableCFO_ = false; return *this; }

  void run(ir::CircuitNode& circuit,
           utils::Logger logger = nullptr) const override;

  void run(ir::CircuitGraphNode& graph,
           utils::Logger logger = nullptr) const override;
};

} // namespace cast