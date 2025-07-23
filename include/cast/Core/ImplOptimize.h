// Implementation file for optimization passes in the CAST simulator.
// This file is separated from the main optimization header to
// 1. Keep the interface clean
// 2. Use in tests

#ifndef CAST_CORE_IMPL_OPTIMIZE_H
#define CAST_CORE_IMPL_OPTIMIZE_H

#include "cast/Core/FusionConfig.h"
#include "cast/Core/IRNode.h"

namespace cast {

namespace impl {

/// @brief Start fusion of gates in the circuit graph.
// In the first stage of fusion, we eagerly collect gates that may possibly
// be fused together, subject to the maximum number of qubits being \c max_k.
// Later we will check if the fusion is beneficial. If not, we reject the
// fusion and put back all these gates to their original positions.
/// @param maxCandidateSize This is different from FusionConfig::maxKOverride.
// This max_k is used to limit the size of the product gate before fusion
// eligibility check.
int startFusion(ir::CircuitGraphNode& graph,
                const FusionConfig* config,
                int maxCandidateSize,
                cast::ir::CircuitGraphNode::row_iterator rowIt,
                int qubit);

int applySizeTwoFusion(ir::CircuitGraphNode& graph, double swapTol);

/// @brief Size-only fusion
/// @param swapTol The tolerance used in swapping analysis. Set to 0.0 or
/// negative to disable swapping analysis.
/// @return Number of fused gates
int applySizeOnlyFusion(ir::CircuitGraphNode& graph, int max_k, double swapTol);

/// @return Number of fused gates
int applyGateFusion(ir::CircuitGraphNode& graph,
                    const FusionConfig* fusionConfig,
                    int maxCandidateSize);

/// @return Number of fused gates
int applyCFOFusion(ir::CircuitNode& circuit,
                   const FusionConfig* config,
                   int maxCandidateSize);

} // namespace impl

} // namespace cast

#endif // CAST_CORE_IMPL_OPTIMIZE_H