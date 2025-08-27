#include "cast/CUDA/CUDAKernelManager.h"

using namespace cast;

int main(int argc, char** argv) {
  auto gate = StandardQuantumGate::H(6);

  CUDAKernelGenConfig config;
  CUDAKernelManager km;

  config.displayInfo(std::cerr);

  km.genStandaloneGate(config, gate, "myGate").consumeError();

  km.emitPTX(llvm::OptimizationLevel::O1);

  km.dumpPTX(std::cerr, "myGate");
}