#ifndef CAST_FPGA_FPGAINSTGEN_H
#define CAST_FPGA_FPGAINSTGEN_H

#include "cast/Core/IRNode.h"
#include "cast/FPGA/FPGAInst.h"

namespace cast::fpga {

struct FPGAInstGenConfig {
public:
  int nLocalQubits;
  int gridSize;

  // If off, apply sequential instruction generation on the default order of
  // blocks present in legacy::CircuitGraph
  bool selectiveGenerationMode;
  int maxUpSize;

  FPGAGateCategoryTolerance tolerances;

  int getNOnChipQubits() const { return nLocalQubits + 2 * gridSize; }
};


// top-level function to generate FPGA instructions from an ir::CircuitGraphNode
std::vector<Instruction> genInstruction(
    const ir::CircuitGraphNode& graph, const FPGAInstGenConfig& config);

} // namespace cast::fpga

#endif // CAST_FPGA_FPGAINSTGEN_H