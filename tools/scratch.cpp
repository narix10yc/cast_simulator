#include "cast/CPU/CPUStatevector.h"
#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "cast/CUDA/Config.h"

using namespace cast;

int main(int argc, char** argv) {

  CUDAKernelManager kernelMgrCUDA;
  CUDAKernelGenConfig cudaGenConfig;
  cudaGenConfig.matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;

  cudaGenConfig.enableTilingGateSize = 999; // disable tiling

  auto gate = StandardQuantumGate::H(0);
  kernelMgrCUDA.genStandaloneGate(cudaGenConfig, gate, "gateImm_0")
      .consumeError();

  kernelMgrCUDA.emitPTX(2, llvm::OptimizationLevel::O1, /* verbose */ 0);
  kernelMgrCUDA.dumpPTX(std::cerr, "gateImm_0");

  return 0;
}