#include "cast/Core/QuantumGate.h"
#include "cast/CPU/StatevectorCPU.h"
#include "cast/CPU/KernelManagerCPU.h"


using namespace cast;

int main(int argc, char** argv) {
  std::cerr << "sizeof(ScalarGateMatrix) = " << sizeof(ScalarGateMatrix) << "\n";
  std::cerr << "sizeof(ComplexSquareMatrix) = " << sizeof(ComplexSquareMatrix) << "\n";
  std::cerr << "sizeof(KrausRep) = " << sizeof(KrausRep) << "\n";
  std::cerr << "sizeof(ChoiRep) = " << sizeof(ChoiRep) << "\n";

  auto quantumGate = StandardQuantumGate::RandomUnitary({0, 3});

  utils::StatevectorCPU<double> sv(20, 2);
  CPUKernelManager kernelMgr;

  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = 1;

  kernelMgr.genCPUGate(kernelGenConfig, quantumGate, "test_gate");
  kernelMgr.initJIT(1, llvm::OptimizationLevel::O1, false, 0);
  kernelMgr.applyCPUKernel(sv.data(), sv.nQubits(), "test_gate");
  // kernelMgr.applyCPUKernelMultithread(sv.data(), sv.nQubits(), "test_gate", 2);


  return 0;
}