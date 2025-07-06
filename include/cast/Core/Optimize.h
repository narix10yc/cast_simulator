#ifndef CAST_CORE_OPTIMIZE_H
#define CAST_CORE_OPTIMIZE_H

#include "cast/Core/IRNode.h"
#include "cast/Core/CostModel.h"
#include "cast/Core/Precision.h"

namespace cast {

struct FusionConfig {
  std::unique_ptr<CostModel> costModel;
  Precision precision;
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

  // Get a size-only fusion configuration. The cost model will be set to null,
  // which means to accept all fusion candidates. 
  static FusionConfig SizeOnly(int max_k);

  std::ostream& display(std::ostream&) const;
};

constexpr int GLOBAL_MAX_K = 7;

/// @brief Optimize the circuit by trying to fuse away all single-qubit gates,
/// including those in if statements.
/// This pass consists of three sub-steps:
/// 1. Eagerly move single-qubit gates inside the then and else blocks.
/// 2. Apply a max-k fusion pass in each block with k=2.
/// 3. Seek for further fusion opportunities by moving gates at the top of the 
/// join block to the bottom of the then and else blocks.
void applyCanonicalizationPass(ir::CircuitNode& circuit, double swaTol = 1e-8);

void applyGateFusion(ir::CircuitGraphNode& graph,
                     const FusionConfig& fusionConfig);

void applyGateFusionPass(ir::CircuitNode& circuit,
                         const FusionConfig& config,
                         bool applyFusionCFOPass = true);
                        
} // namespace cast::ir

#endif // CAST_CORE_OPTIMIZE_H