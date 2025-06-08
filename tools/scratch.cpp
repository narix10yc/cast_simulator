#include "cast/CPU/CPUKernelManager.h"

using namespace cast;

int main() {
  CPUKernelManager kernelMgr;
  CPUKernelGenConfig config;
  config.matrixLoadMode = MatrixLoadMode::StackLoadMatElems;
  config.zeroTol = 0.0;
  config.oneTol = 0.0;

  auto gate = StandardQuantumGate::H(0);
  kernelMgr.genCPUGate(config, gate, "testKernel");

  kernelMgr.dumpIR("testKernel");

  return 0;
}