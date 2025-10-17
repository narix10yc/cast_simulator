#ifndef CAST_FPGA_FPGAINSTGEN_H
#define CAST_FPGA_FPGAINSTGEN_H

#include "cast/Core/IRNode.h"
#include "cast/FPGA/FPGAInst.h"

namespace cast::fpga {

struct FPGAInstGenConfig {
public:
  int nLocalQubits = 14;
  int gridSize = 4;

  // If off, apply sequential instruction generation on the default order of
  // blocks present in the CircuitGraph
  bool selectiveGenerationMode = true;
  int maxUpSize = 5;

  double tolerance = 1e-8;

  int getNOnChipQubits() const { return nLocalQubits + 2 * gridSize; }
};

// top-level function to generate FPGA instructions from an ir::CircuitGraphNode
std::vector<Instruction> genInstruction(const ir::CircuitGraphNode& graph,
                                        const FPGAInstGenConfig& config);

} // namespace cast::fpga

#endif // CAST_FPGA_FPGAINSTGEN_H