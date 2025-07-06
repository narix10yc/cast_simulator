// Implementation file for optimization passes in the CAST simulator.
// This file is separated from the main optimization header to 
// 1. Keep the interface clean
// 2. Use in tests

#ifndef CAST_CORE_IMPL_OPTIMIZE_H
#define CAST_CORE_IMPL_OPTIMIZE_H

#include "cast/Core/IRNode.h"

namespace cast { 

struct FusionConfig;

namespace impl {

/// @brief Start fusion of gates in the circuit graph.
// In the first stage of fusion, we eagerly collect gates that may possibly
// be fused together, subject to the maximum number of qubits being \c max_k.
// Later we will check if the fusion is beneficial. If not, we reject the
// fusion and put back all these gates to their original positions.
/// @param max_k_candidate This is different from FusionConfig::maxKOverride.
// This max_k is used to limit the size of the product gate before fusion 
// eligibility check.
int startFusion(cast::ir::CircuitGraphNode& graph,
                const FusionConfig& config,
                int max_k_candidate,
                cast::ir::CircuitGraphNode::row_iterator rowIt,
                int qubit);

/// @brief Size-only fusion
/// @param swaTol The tolerance used in swapping analysis. Set to 0.0 or 
/// negative to disable swapping analysis.
/// @return Number of fused gates
int applySizeOnlyFusion(cast::ir::CircuitGraphNode& graph,
                        int max_k,
                        double swaTol);

bool applyFusionCFOPass(ir::CircuitNode& circuit, const FusionConfig& config);

                        
} // namespace impl


} // namespace cast

#endif // CAST_CORE_IMPL_OPTIMIZE_H