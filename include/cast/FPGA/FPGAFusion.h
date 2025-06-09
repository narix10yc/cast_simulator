#ifndef CAST_FPGA_FPGAFUSION_H
#define CAST_FPGA_FPGAFUSION_H

#include "cast/FPGA/FPGAConfig.h"
#include "cast/IR/IRNode.h"

namespace cast::fpga {

void applyFPGAGateFusion(cast::ir::CircuitGraphNode& graph,
                         const FPGAFusionConfig& config);

} // namespace cast::fpga

#endif // CAST_FPGA_FPGAFUSION_H