#include "cast/Core/QuantumGate.h"
#include "cast/CPU/StatevectorCPU.h"
#include "cast/CPU/KernelManagerCPU.h"


using namespace cast;

int main(int argc, char** argv) {
  std::cerr << "sizeof(ScalarGateMatrix) = " << sizeof(ScalarGateMatrix) << "\n";
  std::cerr << "sizeof(ComplexSquareMatrix) = " << sizeof(ComplexSquareMatrix) << "\n";
  std::cerr << "sizeof(KrausRep) = " << sizeof(KrausRep) << "\n";
  std::cerr << "sizeof(ChoiRep) = " << sizeof(ChoiRep) << "\n";

  int simd_s = 1;
  // auto quantumGate = StandardQuantumGate::RandomUnitary({1});
  auto quantumGate = StandardQuantumGate::Create(
    ScalarGateMatrix::H(), nullptr, {1});

  utils::StatevectorCPU<double> sv(3, simd_s);
  sv.randomize();
  sv.print();
  CPUKernelManager kernelMgr;

  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = simd_s;
  kernelGenConfig.matrixLoadMode = MatrixLoadMode::UseMatImmValues;
  kernelGenConfig.precision = 64;

  kernelMgr.genCPUGate(kernelGenConfig, quantumGate, "test_gate");
  kernelMgr.initJIT(1, llvm::OptimizationLevel::O1, false, 0);
  kernelMgr.applyCPUKernel(sv.data(), sv.nQubits() - 1, "test_gate");

  std::cerr << CYAN("Single-thread successfully returned") << "\n";
  sv.print(std::cerr);

  // kernelMgr.applyCPUKernelMultithread(sv.data(), sv.nQubits() - 1, "test_gate", 2);

  // std::cerr << CYAN("Multi-thread successfully returned") << "\n";

  return 0;
}