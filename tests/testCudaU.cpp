#include "simulation/KernelManager.h"
#include "tests/TestKit.h"
#include "simulation/StatevectorCPU.h"
#include "simulation/StatevectorCUDA.h"
#include <random>

using namespace cast;
using namespace cast::test;

template<unsigned nQubits>
static void f() {
  test::TestSuite suite(
    "Gate U1q (" + std::to_string(nQubits) + " qubits)");

  // kernel manager must be declared before statevector due to the order they
  // are destructed.
  CUDAKernelManager kernelMgrCUDA;
  utils::StatevectorCPU<double> svCPU(nQubits, /* simd_s */ 0);
  utils::StatevectorCUDA<double> svCUDA0(nQubits), svCUDA1(nQubits);

  const auto randomizeSV = [&]() {
    svCUDA0.randomize();
    svCUDA1 = svCUDA0;
    cudaMemcpy(svCPU.data(), svCUDA0.dData(), svCUDA0.sizeInBytes(),
      cudaMemcpyDeviceToHost);
  };

  // generate random unitary gates
  std::vector<std::shared_ptr<LegacyQuantumGate>> gates;
  gates.reserve(nQubits);
  for (int q = 0; q < nQubits; q++) {
    gates.emplace_back(
      std::make_shared<LegacyQuantumGate>(LegacyQuantumGate::RandomUnitary(q)));
  }

  CUDAKernelGenConfig cudaGenConfig;
  cudaGenConfig.matrixLoadMode = CUDAKernelGenConfig::UseMatImmValues;
  for (int q = 0; q < nQubits; q++) {
    kernelMgrCUDA.genCUDAGate(
      cudaGenConfig, gates[q], "gateImm_" + std::to_string(q));
  }

  // cudaGenConfig.forceDenseKernel = true;
  // cudaGenConfig.matrixLoadMode = CUDAKernelGenConfig::LoadInConstMemSpace;
  // for (int q = 0; q < nQubits; q++) {
  //   kernelMgrCUDA.genCUDAGate(
  //     cudaGenConfig, gates[q], "gateConstMemSpace_" + std::to_string(q));
  // }

  kernelMgrCUDA.emitPTX(2, llvm::OptimizationLevel::O1, /* verbose */ 0);
  kernelMgrCUDA.initCUJIT(2, /* verbose */ 0);
  for (unsigned i = 0; i < 2; i++) {
    randomizeSV();
    std::stringstream ss;
    auto qubit = gates[i]->qubits[0];
    ss << "Apply U1q at " << qubit << ": ";
    // auto immFuncName = "gateImm_" + std::to_string(i);
    // auto loadFuncName = "gateConstMemSpace_" + std::to_string(i);
    kernelMgrCUDA.launchCUDAKernel(
      svCUDA0.dData(), svCUDA0.nQubits(), kernelMgrCUDA.kernels()[i]);
    suite.assertClose(svCUDA0.norm(), 1.0,
      ss.str() + "CUDA SV norm equals to 1", GET_INFO());

    svCPU.applyGate(*gates[i]);
    suite.assertClose(svCUDA0.prob(qubit), svCPU.prob(qubit),
      ss.str() + "CUDA and CPU SV prob match", GET_INFO());
    // suite.assertClose(sv1.norm(), 1.0, ss.str() + ": Load Norm", GET_INFO());
    // suite.assertClose(utils::fidelity(sv0, sv2), 1.0,
    //   ss.str() + ": Imm Fidelity", GET_INFO());
    // suite.assertClose(utils::fidelity(sv1, sv2), 1.0,
    //   ss.str() + ": Load Fidelity", GET_INFO());
  }
  suite.displayResult();
}

void test::test_cudaU() {
  f<8>();
  f<12>();
}
