#ifndef CAST_CORE_OPTIMIZER_H
#define CAST_CORE_OPTIMIZER_H

#include "cast/Core/FusionConfig.h"
#include "cast/Core/IRNode.h"
#include "utils/Logger.h"

namespace cast {

// The base class for optimizers.
class OptimizerBase {
protected:
  // Default is SizeOnlyFusionConfig with size 3.
  std::unique_ptr<FusionConfig> fusionConfig_;

  bool enableCanonicalization_ = true;
  bool enableFusion_ = true;
  bool enableCFO_ = false;

public:
  virtual ~OptimizerBase() = default;

  void run(ir::CircuitNode& circuit, utils::Logger logger = nullptr) const;

  void run(ir::CircuitGraphNode& graph, utils::Logger logger = nullptr) const;
}; // OptimizerBase

// We use CRTP to allow method chaining.
template <typename Derived> class Optimizer : public OptimizerBase {
public:
  // Default fusion config is SizeOnlyFusionConfig with size 3.
  Optimizer() { fusionConfig_ = std::make_unique<SizeOnlyFusionConfig>(3); }

  Derived& setFusionConfig(std::unique_ptr<FusionConfig> cfg) {
    fusionConfig_ = std::move(cfg);
    return static_cast<Derived&>(*this);
  }

  const FusionConfig* getFusionConfig() const { return fusionConfig_.get(); }
  FusionConfig* getFusionConfig() { return fusionConfig_.get(); }

  // Size-only fusion does not take any queries.
  Derived& setSizeOnlyFusionConfig(int size) {
    fusionConfig_ = std::make_unique<SizeOnlyFusionConfig>(size);
    return static_cast<Derived&>(*this);
  }

  Derived& enableCanonicalization(bool enable = true) {
    enableCanonicalization_ = enable;
    return static_cast<Derived&>(*this);
  }

  Derived& enableFusion(bool enable = true) {
    enableFusion_ = enable;
    return static_cast<Derived&>(*this);
  }

  Derived& enableCFO(bool enable = true) {
    enableCFO_ = enable;
    return static_cast<Derived&>(*this);
  }

}; // class Optimizer

} // namespace cast

#endif // CAST_CORE_OPTIMIZER_H