#ifndef CAST_CORE_OPTIMIZE_H
#define CAST_CORE_OPTIMIZE_H

#include "cast/Core/IRNode.h"
#include "cast/Core/CostModel.h"

namespace cast {

struct FusionConfig {
  int precision;
  int nThreads;

  double zeroTol;
  bool incrementScheme;
  // the maximum number of qubits in fused gates
  int maxKOverride;
  /// How much benefit do we recognize as significant. For example, if set to
  /// 0.1, then we accept fusion whenever costModel predicts >10% improvement
  double benefitMargin;

  static const FusionConfig Minor;
  static const FusionConfig Default;
  static const FusionConfig Aggressive;

  std::ostream& display(std::ostream&) const;
};

constexpr int GLOBAL_MAX_K = 7;

int startFusion(cast::ir::CircuitGraphNode& graph,
                const FusionConfig& config,
                const CostModel* costModel,
                int cur_max_k,
                cast::ir::CircuitGraphNode::row_iterator rowIt,
                int qubit);

/// @brief Optimize the circuit by trying to fuse away all single-qubit gates,
/// including those in if statements.
/// This pass consists of three sub-steps:
/// 1. Eagerly move single-qubit gates inside the then and else blocks.
/// 2. Apply a max-k fusion pass in each block with k=2.
/// 3. Seek for further fusion opportunities by moving gates at the top of the 
/// join block to the bottom of the then and else blocks.
void applyPreFusionCFOPass(ir::CircuitNode& circuit);

void applyGateFusion(ir::CircuitGraphNode& graph,
                     const FusionConfig& fusionConfig,
                     const CostModel* costModel);

void applyGateFusion(ir::CircuitNode& circuit,
                     const FusionConfig& config,
                     const CostModel* costModel);
                        
} // namespace cast::ir

#endif // CAST_CORE_OPTIMIZE_H