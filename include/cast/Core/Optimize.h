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
  // Swapping tolerance. Set to 0.0 or negative to disable.
  double swaTol;
  bool incrementScheme;
  bool multiTraversal;
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
void applyCanonicalizationPass(ir::CircuitNode& circuit, double swaTol = 0.0);

void applyGateFusionPass(ir::CircuitNode& circuit,
                         const FusionConfig& config,
                         bool applyFusionCFOPass = true);
                        
} // namespace cast::ir

#endif // CAST_CORE_OPTIMIZE_H