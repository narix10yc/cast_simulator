#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "cast/CUDA/Config.h"

using namespace cast;

int main(int argc, char** argv) {
  CUDAKernelManager kernelMgr;

  CUDAKernelGenConfig config;
  // disable tiling for testing
  config.enableTilingGateSize = 999;
  // config.matrixLoadMode = CUDAMatrixLoadMode::LoadInConstMemSpace;

  config.displayInfo(std::cerr) << "\n";

  auto rst = kernelMgr.genStandaloneGate(
      config, StandardQuantumGate::H(0), "testKernel");

  if (!rst) {
    std::cerr << "Error generating standalone gate: " << rst.takeError()
              << "\n";
    return 1;
  }

  kernelMgr.emitPTX(1, llvm::OptimizationLevel::O1);
  kernelMgr.initCUJIT(1, 1);

  std::cerr << kernelMgr.getPTXString(0) << "\n";

  return 0;
}