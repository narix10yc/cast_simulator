#include "cast/CPU/CPUStatevector.h"
#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "tests/TestKit.h"

using namespace cast;
using namespace cast::test;

template <unsigned nQubits> static void f() {
  test::TestSuite suite("Gate U1q (" + std::to_string(nQubits) + " qubits)");

  // kernel manager must be declared before statevector due to the order they
  // are destructed.
  CUDAKernelManager kernelMgrCUDA;
  // we have to use W0 here to allow memcpy from cuda sv to cpu sv
  cast::CPUStatevector<double> svCPU(nQubits, cast::CPUSimdWidth::W0);
  cast::CUDAStatevector<double> svCUDA0(nQubits), svCUDA1(nQubits);
  svCUDA0.initialize();
  svCUDA1.initialize();

  // const auto randomizeSV = [&]() {
  //   svCUDA0.randomize();
  //   svCUDA1 = svCUDA0;
  //   cudaMemcpy(svCPU.data(),
  //              svCUDA0.dData(),
  //              svCUDA0.sizeInBytes(),
  //              cudaMemcpyDeviceToHost);
  // };

  const auto randomizeSV = [&]() {
    std::fill_n(svCPU.data(), svCPU.size(), 1.0);
    svCPU.normalize();
    cudaMemcpy(svCUDA0.dData(), svCPU.data(), svCUDA0.sizeInBytes(),
               cudaMemcpyHostToDevice);
    svCUDA1 = svCUDA0;
  };

  // generate random unitary gates
  std::vector<QuantumGatePtr> gates;
  gates.reserve(nQubits);
  for (int q = 0; q < nQubits; q++) {
    // gates.emplace_back(StandardQuantumGate::RandomUnitary(q));
    gates.emplace_back(StandardQuantumGate::H(q));
  }

  CUDAKernelGenConfig cudaGenConfig;
  cudaGenConfig.matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;
  // cudaGenConfig.forceDenseKernel = true;
  cudaGenConfig.enableTilingGateSize = 9999; // enable tiling
  cudaGenConfig.precision = Precision::F64;

  for (int q = 0; q < nQubits; q++) {
    auto rst = kernelMgrCUDA.genStandaloneGate(
        cudaGenConfig, gates[q], "gateImm_" + std::to_string(q));
    if (!rst) {
      suite.assertFalse("Failed to generate CUDA kernel for gate " +
                            std::to_string(q) + ": " + rst.takeError(),
                        GET_INFO());
    }
  }

  // cudaGenConfig.forceDenseKernel = true;
  // cudaGenConfig.matrixLoadMode = CUDAKernelGenConfig::LoadInConstMemSpace;
  // for (int q = 0; q < nQubits; q++) {
  //   kernelMgrCUDA.genCUDAGate(
  //     cudaGenConfig, gates[q], "gateConstMemSpace_" + std::to_string(q));
  // }

  kernelMgrCUDA.emitPTX(2, llvm::OptimizationLevel::O1, /* verbose */ 0);

  kernelMgrCUDA.dumpPTX(std::cerr, "gateImm_0");

  kernelMgrCUDA.initCUJIT(2, /* verbose */ 0);
  for (unsigned q = 0; q < nQubits; q++) {
    std::stringstream ss;
    assert(q == gates[q]->qubits()[0]);
    ss << "Apply U1q at " << q << ": ";

    randomizeSV();
    suite.assertCloseF64(svCUDA0.prob(q),
                         svCPU.prob(q),
                         ss.str() + "Prob match before applying gate",
                         GET_INFO());

    // Apply CPU gate
    svCPU.applyGate(*llvm::dyn_cast<StandardQuantumGate>(gates[q].get()));

    // Apply CUDA gate
    auto* kernelInfo =
        kernelMgrCUDA.getKernelByName("gateImm_" + std::to_string(q));
    assert(kernelInfo != nullptr);
    kernelMgrCUDA.launchCUDAKernel(
        svCUDA0.dData(), svCUDA0.nQubits(), *kernelInfo);

    svCPU.display(std::cerr << "CPU SV after applying gate " << q << "\n")
        << "\n";

    svCUDA0.display(std::cerr << "CUDA SV after applying gate " << q << "\n")
        << "\n";

    suite.assertCloseF64(
        svCUDA0.norm(), 1.0, ss.str() + "CUDA SV norm equals to 1", GET_INFO());
    suite.assertCloseF64(svCUDA0.prob(q),
                         svCPU.prob(q),
                         ss.str() + "Prob match after applying gate",
                         GET_INFO());
  }
  suite.displayResult();
}

void test::test_cudaU() {
  f<4>();
  // f<8>();
  // f<12>();
}
