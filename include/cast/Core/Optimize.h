#ifndef CAST_CORE_OPTIMIZE_H
#define CAST_CORE_OPTIMIZE_H

#include "cast/Core/IRNode.h"
#include "cast/Core/CostModel.h"
#include "cast/Core/FusionConfig.h"

namespace cast {

class Optimizer {
  std::vector<std::function<void(ir::CircuitNode&)>> passes;
public:
  void addPass(std::function<void(ir::CircuitNode&)> pass) {
    passes.push_back(std::move(pass));
  }

  void run(ir::CircuitNode& circuit) {
    for (auto& pass : passes) {
      pass(circuit);
    }
  }
}; // class Optimizer


/// @brief Optimize the circuit by trying to fuse away all single-qubit gates,
/// including those in if statements.
void applyCanonicalizationPass(ir::CircuitNode& circuit, double swapTol = 0.0);

void applyGateFusionPass(ir::CircuitNode& circuit, const FusionConfig* config);
                        
} // namespace cast::ir

#endif // CAST_CORE_OPTIMIZE_H