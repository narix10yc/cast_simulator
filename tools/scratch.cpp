#include "cast/Core/QuantumGate.h"
#include "cast/CPU/StatevectorCPU.h"
#include "cast/CPU/KernelManagerCPU.h"
#include "cast/IR/IRNode.h"
#include "cast/CostModel.h"
#include "cast/Fusion.h"


using namespace cast;

int main(int argc, char** argv) {
  // std::cerr << "sizeof(ScalarGateMatrix) = " << sizeof(ScalarGateMatrix) << "\n";
  // std::cerr << "sizeof(ComplexSquareMatrix) = " << sizeof(ComplexSquareMatrix) << "\n";
  // std::cerr << "sizeof(KrausRep) = " << sizeof(KrausRep) << "\n";
  // std::cerr << "sizeof(ChoiRep) = " << sizeof(ChoiRep) << "\n";

  // int simd_s = 1;
  // // auto quantumGate = StandardQuantumGate::RandomUnitary({1});
  // auto quantumGate = StandardQuantumGate::Create(
  //   ScalarGateMatrix::H(), nullptr, {1});

  // utils::StatevectorCPU<double> sv(3, simd_s);
  // sv.randomize();
  // sv.print();
  // CPUKernelManager kernelMgr;

  // CPUKernelGenConfig kernelGenConfig;
  // kernelGenConfig.simd_s = simd_s;
  // kernelGenConfig.matrixLoadMode = MatrixLoadMode::UseMatImmValues;
  // kernelGenConfig.precision = 64;

  // kernelMgr.genCPUGate(kernelGenConfig, quantumGate, "test_gate");
  // kernelMgr.initJIT(1, llvm::OptimizationLevel::O1, false, 0);
  // kernelMgr.applyCPUKernel(sv.data(), sv.nQubits() - 1, "test_gate");

  // std::cerr << CYAN("Single-thread successfully returned") << "\n";
  // sv.print(std::cerr);

  // kernelMgr.applyCPUKernelMultithread(sv.data(), sv.nQubits() - 1, "test_gate", 2);

  // std::cerr << CYAN("Multi-thread successfully returned") << "\n";


  auto gate0 = StandardQuantumGate::RandomUnitary({0});
  auto gate1 = StandardQuantumGate::RandomUnitary({0});
  // auto gate1 = StandardQuantumGate::Create(
    // ScalarGateMatrix::I1(), nullptr, {0});
  gate0->displayInfo(std::cerr << "Gate 0", 3) << "\n";
  gate1->displayInfo(std::cerr << "Gate 1", 3) << "\n";

  ir::CircuitGraphNode graph;
  graph.insertGate(gate0);
  graph.insertGate(gate1);

  CPUKernelManager kernelMgr;
  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = 1;
  kernelGenConfig.precision = 64;

  auto allGatesBeforeFusion = graph.getAllGatesShared();
  for (int q = 0; q < allGatesBeforeFusion.size(); ++q) {
    kernelMgr.genCPUGate(kernelGenConfig, allGatesBeforeFusion[q],
      "before" + std::to_string(q));
  }

  NaiveCostModel naiveCostMode(3, -1, 0);
  auto fusionConfig = FusionConfig::Aggressive;
  cast::applyGateFusion(fusionConfig, &naiveCostMode, graph);

  graph.visualize(std::cerr) << "\n";

  auto allGatesAfterFusion = graph.getAllGatesShared();
  allGatesAfterFusion[0]->displayInfo(std::cerr << "Gate Fused", 3) << "\n";
  for (int q = 0; q < allGatesAfterFusion.size(); ++q) {
    kernelMgr.genCPUGate(kernelGenConfig, allGatesAfterFusion[q],
      "after" + std::to_string(q));
  }
  
  kernelMgr.initJIT(1, llvm::OptimizationLevel::O1, false, 0);
  
  utils::StatevectorCPU<double> sv0(4, 1), sv1(4, 1);
  sv0.randomize();
  sv1 = sv0;
  sv0.print(std::cerr) << "\n";

  for (int q = 0; q < allGatesBeforeFusion.size(); ++q) {
    kernelMgr.applyCPUKernel(sv0.data(), sv0.nQubits(), "before" + std::to_string(q));
  }
  for (int q = 0; q < allGatesAfterFusion.size(); ++q) {
    kernelMgr.applyCPUKernel(sv1.data(), sv1.nQubits(), "after" + std::to_string(q));
  }

  sv0.print(std::cerr) << "\n";
  sv1.print(std::cerr) << "\n";

  std::cerr << "Fidelity = "
            << utils::fidelity(sv0, sv1) << "\n";

  // ComplexSquareMatrix mat0 {
  // // real
  // {1.0, 2.0, 3.0, 4.0},
  // // imag
  // {0.0, 0.0, 0.0, 0.0}
  // };
  // QuantumGatePtr gate0 = StandardQuantumGate::Create(
  //   std::make_shared<ScalarGateMatrix>(mat0), nullptr, {0});
  // QuantumGatePtr gate1 = gate0;
  // auto gate = cast::matmul(gate0.get(), gate1.get());
  // auto* stdQuGate = llvm::dyn_cast<StandardQuantumGate>(gate.get());
  // stdQuGate->displayInfo(std::cerr, 3) << "\n";

  return 0;
}