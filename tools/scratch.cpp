#include "cast/Core/QuantumGate.h"
#include "cast/CPU/CPUDensityMatrix.h"
#include "cast/CPU/CPUKernelManager.h"
#include "cast/IR/IRNode.h"
#include "cast/CostModel.h"
#include "cast/Fusion.h"

using namespace cast;

int main(int argc, char** argv) {

  // a SPC(0.1) noise channel
  auto gate = StandardQuantumGate::I1(0);
  gate->setNoiseSPC(0.1);

  cast::getSuperopGate(gate)->displayInfo(std::cerr, 3);

  CPUKernelManager kernelMgr;
  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = 1;

  kernelMgr.genCPUGate(kernelGenConfig, gate, "I1_0");
  kernelMgr.initJIT(1, llvm::OptimizationLevel::O1, false);

  CPUDensityMatrix<double> dm(3, 1);
  dm.initialize();
  kernelMgr.applyCPUKernel(dm.data(), dm.nQubits(), "I1_0");

  dm.print(std::cerr);

  return 0;
}