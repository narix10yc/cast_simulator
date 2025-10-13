#include "cast/CPU/CPUStatevector.h"
#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "tests/TestKit.h"

using namespace cast;
using namespace cast::test;

template <unsigned nQubits> static void f() {
  test::TestSuite suite("Gate U1q (" + std::to_string(nQubits) + " qubits)");

  // we have to use W0 here to allow memcpy from cuda sv to cpu sv
  cast::CPUStatevector<double> svCPU(nQubits, cast::CPUSimdWidth::W0);
  cast::CUDAStatevector<double> svCUDA(nQubits);

  // use 2 worker threads
  CUDAKernelManager km(2);

  const auto randomizeSV = [&]() {
    svCUDA.randomize();
    cudaMemcpy(svCPU.data(),
               svCUDA.dData(),
               svCUDA.sizeInBytes(),
               cudaMemcpyDeviceToHost);
  };

  // generate random unitary gates
  std::vector<QuantumGatePtr> gates;
  gates.reserve(nQubits);
  for (unsigned q = 0; q < nQubits; q++)
    gates.emplace_back(StandardQuantumGate::RandomUnitary(q));

  CUDAKernelGenConfig cfg;
  cfg.matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;
  cfg.precision = Precision::FP64;

  for (unsigned q = 0; q < nQubits; q++)
    CHECK(suite, km.genGate(cfg, gates[q], "gateImm_" + std::to_string(q)));

  for (unsigned q = 0; q < nQubits; q++) {
    std::stringstream ss;
    assert(q == gates[q]->qubits()[0]);
    ss << "Apply U1q at " << q << ": ";

    randomizeSV();
    suite.assertCloseF64(svCUDA.prob(q),
                         svCPU.prob(q),
                         ss.str() + "Prob match before applying gate",
                         GET_INFO());

    // Apply CPU gate
    svCPU.applyGate(*llvm::dyn_cast<StandardQuantumGate>(gates[q].get()));

    // Apply CUDA gate
    auto* kernelInfo = km.getKernelByName("gateImm_" + std::to_string(q));
    assert(kernelInfo != nullptr);
    km.setLaunchConfig(svCUDA.getDevicePtr(), svCUDA.nQubits());
    km.enqueueKernelLaunch(*kernelInfo);
    km.syncKernelExecution();

    suite.assertCloseF64(
        svCUDA.norm(), 1.0, ss.str() + "CUDA SV norm equals to 1", GET_INFO());
    suite.assertCloseF64(svCUDA.prob(q),
                         svCPU.prob(q),
                         ss.str() + "Prob match after applying gate",
                         GET_INFO());
  }
  suite.displayResult();
}

void test::test_cudaU() {
  f<4>();
  f<8>();
  f<12>();
}
