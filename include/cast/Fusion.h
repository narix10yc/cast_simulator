#ifndef CAST_CPUFUSION_H
#define CAST_CPUFUSION_H

#include "cast/CostModel.h"
#include "cast/Legacy/FPGAConfig.h"
#include "cast/IR/IRNode.h"
#include <cassert>

namespace cast {
  namespace legacy {
    class CircuitGraph;
    class QuantumGate;
  } // namespace legacy

struct FusionConfig {
  int precision;
  int nThreads;

  double zeroTol;
  bool multiTraverse;
  bool incrementScheme;
  /// How much benefit do we recognize as significant. For example, if set to
  /// 0.1, then we accept fusion whenever costModel predicts >10% improvement
  double benefitMargin;

  static FusionConfig Preset(int level) {
    if (level == 0)
      return Disable;
    if (level == 1)
      return Minor;
    if (level == 2)
      return Default;
    if (level == 3)
      return Aggressive;
    assert(false && "Unknown CPUFusionConfig Preset");
    return Disable;
  }

  static const FusionConfig Disable;
  static const FusionConfig Minor;
  static const FusionConfig Default;
  static const FusionConfig Aggressive;

  std::ostream& display(std::ostream&) const;
};

void applyGateFusion(
    const FusionConfig&, const CostModel*, legacy::CircuitGraph&, int max_k=7);

void applyGateFusion(
    const FusionConfig&, const CostModel*, ir::CircuitGraphNode&, int max_k=7);

void applyFPGAGateFusion(legacy::CircuitGraph& graph,
                         const legacy::fpga::FPGAFusionConfig& config);

} // namespace cast

#endif // CAST_CPUFUSION_H