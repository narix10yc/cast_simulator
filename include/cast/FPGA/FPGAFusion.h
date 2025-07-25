#ifndef CAST_FPGA_FPGAFUSION_H
#define CAST_FPGA_FPGAFUSION_H

#include "cast/Core/IRNode.h"
#include "cast/FPGA/FPGAGateCategory.h"

namespace cast::fpga {

struct FPGAFusionConfig {
  int maxUnitaryPermutationSize;
  bool ignoreSingleQubitNonCompGates;
  bool multiTraverse;
  double tolerance;

  static const FPGAFusionConfig Default;
};

void applyFPGAGateFusion(cast::ir::CircuitGraphNode& graph,
                         const FPGAFusionConfig& config);

} // namespace cast::fpga

#endif // CAST_FPGA_FPGAFUSION_H