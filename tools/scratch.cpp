#include "cast/CUDA/CUDAKernelManager.h"

using namespace cast;

int main(int argc, char** argv) {

  QuantumGatePtr qGate = StandardQuantumGate::RandomUnitary({0, 2});

  CUDAKernelManager kernelMgr;
  CUDAKernelGenConfig config;
  config.displayInfo(std::cerr);

  auto r = kernelMgr.genStandaloneGate(config, qGate, "test_gate");
  if (!r) {
    std::cerr << "[Error] Failed to generate standalone gate.\n";
    return 1;
  }

  kernelMgr.emitPTX(1, llvm::OptimizationLevel::O1, 1);
  kernelMgr.dumpPTX(std::cerr, "test_gate");

  kernelMgr.initCUJIT(1, 1);

  return 0;
}