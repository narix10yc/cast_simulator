#ifndef CAST_CORE_OPTIMIZER_H
#define CAST_CORE_OPTIMIZER_H

#include "cast/Core/IRNode.h"
#include "cast/Core/FusionConfig.h"
#include "utils/Logger.h"

namespace cast {

class Optimizer {
public:
  virtual ~Optimizer() = default;

  virtual void run(ir::CircuitNode& circuit, utils::Logger logger) const = 0;

  virtual void run(ir::CircuitGraphNode& graph, utils::Logger logger) const = 0;

  virtual std::ostream& displayInfo(std::ostream& os, int verbose = 1) const {
    return os << "Optimizer @ " << this << "\n";
  }

}; // class Optimizer

/// @brief Optimize the circuit by trying to fuse away all single-qubit gates,
/// including those in if statements.
void applyCanonicalizationPass(ir::CircuitNode& circuit, double swapTol);

void applyGateFusionPass(ir::CircuitNode& circuit, const FusionConfig* config);
                        
} // namespace cast::ir

#endif // CAST_CORE_OPTIMIZER_H