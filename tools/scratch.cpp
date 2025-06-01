#include "cast/Core/QuantumGate.h"
#include "cast/CPU/CPUStatevector.h"
#include "cast/CPU/CPUKernelManager.h"
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

  // cast::CPUStatevector<double> sv(3, simd_s);
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


  // auto gate0 = StandardQuantumGate::RandomUnitary({0});
  // auto gate1 = StandardQuantumGate::RandomUnitary({0});
  // // auto gate1 = StandardQuantumGate::Create(
  //   // ScalarGateMatrix::I1(), nullptr, {0});
  // gate0->displayInfo(std::cerr << "Gate 0", 3) << "\n";
  // gate1->displayInfo(std::cerr << "Gate 1", 3) << "\n";

  // ir::CircuitGraphNode graph;
  // graph.insertGate(gate0);
  // graph.insertGate(gate1);

  // CPUKernelManager kernelMgr;
  // CPUKernelGenConfig kernelGenConfig;
  // kernelGenConfig.simd_s = 1;
  // kernelGenConfig.precision = 64;

  // auto allGatesBeforeFusion = graph.getAllGatesShared();
  // for (int q = 0; q < allGatesBeforeFusion.size(); ++q) {
  //   kernelMgr.genCPUGate(kernelGenConfig, allGatesBeforeFusion[q],
  //     "before" + std::to_string(q));
  // }

  // NaiveCostModel naiveCostMode(3, -1, 0);
  // auto fusionConfig = FusionConfig::Aggressive;
  // cast::applyGateFusion(fusionConfig, &naiveCostMode, graph);

  // graph.visualize(std::cerr) << "\n";

  // auto allGatesAfterFusion = graph.getAllGatesShared();
  // allGatesAfterFusion[0]->displayInfo(std::cerr << "Gate Fused", 3) << "\n";
  // for (int q = 0; q < allGatesAfterFusion.size(); ++q) {
  //   kernelMgr.genCPUGate(kernelGenConfig, allGatesAfterFusion[q],
  //     "after" + std::to_string(q));
  // }
  
  // kernelMgr.initJIT(1, llvm::OptimizationLevel::O1, false, 0);
  
  // cast::CPUStatevector<double> sv0(4, 1), sv1(4, 1);
  // sv0.randomize();
  // sv1 = sv0;
  // sv0.print(std::cerr) << "\n";

  // for (int q = 0; q < allGatesBeforeFusion.size(); ++q) {
  //   kernelMgr.applyCPUKernel(sv0.data(), sv0.nQubits(), "before" + std::to_string(q));
  // }
  // for (int q = 0; q < allGatesAfterFusion.size(); ++q) {
  //   kernelMgr.applyCPUKernel(sv1.data(), sv1.nQubits(), "after" + std::to_string(q));
  // }

  // sv0.print(std::cerr) << "\n";
  // sv1.print(std::cerr) << "\n";

  // std::cerr << "Fidelity = "
  //           << cast::fidelity(sv0, sv1) << "\n";

  auto gate0 = StandardQuantumGate::Create(ScalarGateMatrix::I1(), nullptr, {0});
  auto gate1 = StandardQuantumGate::Create(ScalarGateMatrix::I1(), nullptr, {1});
  auto gate = StandardQuantumGate::Create(ScalarGateMatrix::I2(), nullptr, {0, 1});
  
  gate0->displayInfo(std::cerr << "Gate 0", 3) << "\n";
  gate1->displayInfo(std::cerr << "Gate 1", 3) << "\n";
  gate->displayInfo(std::cerr << "Gate", 3) << "\n";

  auto prod = cast::matmul(gate0.get(), gate1.get());
  auto* stdQuGate = llvm::dyn_cast<StandardQuantumGate>(prod.get());
  assert(stdQuGate != nullptr);
  assert(stdQuGate->getScalarGM() != nullptr);
  assert(gate0->getScalarGM() != nullptr);
  assert(gate1->getScalarGM() != nullptr);
  assert(gate->getScalarGM() != nullptr);
  prod->displayInfo(std::cerr << "Product", 3) << "\n";
  std::cerr << "Maximum norm = "
            << cast::maximum_norm(
                 stdQuGate->getScalarGM()->matrix(),
                 gate->getScalarGM()->matrix())
            << "\n";

  return 0;
}