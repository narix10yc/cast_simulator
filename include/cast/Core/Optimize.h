#ifndef CAST_CORE_OPTIMIZE_H
#define CAST_CORE_OPTIMIZE_H

#include "cast/Core/IRNode.h"
#include "cast/Core/CostModel.h"

namespace cast {

struct FusionConfig {
  int precision;
  int nThreads;

  double zeroTol;
  bool multiTraverse;
  bool incrementScheme;
  /// How much benefit do we recognize as significant. For example, if set to
  /// 0.1, then we accept fusion whenever costModel predicts >10% improvement
  double benefitMargin;

  static const FusionConfig Disable;
  static const FusionConfig Minor;
  static const FusionConfig Default;
  static const FusionConfig Aggressive;

  std::ostream& display(std::ostream&) const;
};

int startFusion(cast::ir::CircuitGraphNode& graph,
                const FusionConfig& config,
                const CostModel* costModel,
                int max_k, 
                cast::ir::CircuitGraphNode::row_iterator rowIt,
                int qubit);

// TODO: We might want to change this function to
// ir::CircuitGraphNode::applyFusion()
void applyGateFusion(cast::ir::CircuitGraphNode& graph,
                     const FusionConfig& fusionConfig,
                     const CostModel* costModel);

void optimize(ir::CircuitNode& circuit,
              const FusionConfig& config,
              const CostModel* costModel);

} // namespace cast::ir

#endif // CAST_CORE_OPTIMIZE_H