#include "cast/CPU/CPUKernelManager.h"
#include "cast/CPU/CPUStatevector.h"

using namespace cast;

int main() {
  // Initialize the CPU kernel manager
  CPUKernelManager kernelManager;

  auto gate = StandardQuantumGate::RandomUnitary({3, 1});
  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.precision = 32;
  kernelGenConfig.simdWidth = 2;

  kernelManager.genCPUGate(kernelGenConfig, gate, "my_gate");
  // kernelManager.initJIT(/* nThreads */ 1, llvm::OptimizationLevel::O1);

  // kernelManager.dumpAsm("my_gate", llvm::errs());
  std::error_code ec;
  llvm::raw_fd_ostream os("dumped.ll", ec);
  kernelManager.dumpIR("my_gate", os);

  // CPUStatevector<float> sv(12, kernelGenConfig.simd_s);
  // sv.randomize();
  // kernelManager.applyCPUKernelMultithread(sv.data(), sv.nQubits(), "my_gate", 8);

  return 0;
}