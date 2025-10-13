#include "cast/CPU/CPUStatevector.h"
#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "tests/TestKit.h"

using namespace cast;
using namespace cast::test;

namespace {

template <int nQubits> void runU2qTest() {
  TestSuite suite("Gate U2q (" + std::to_string(nQubits) + " qubits)");

  // we have to use W0 here to allow direct memcpy between host and device
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

  std::vector<QuantumGatePtr> gates;
  gates.reserve(nQubits);
  for (int q = 0; q < nQubits; q += 2) {
    QuantumGate::TargetQubitsType qubits;
    utils::sampleNoReplacement(nQubits, 2, qubits);
    gates.emplace_back(StandardQuantumGate::RandomUnitary(qubits));
  }

  CUDAKernelGenConfig cgCfg;
  cgCfg.matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;
  for (std::size_t i = 0; i < gates.size(); ++i) {
    llvm::cantFail(
        km.genGate(cgCfg, gates[i], "gateImm_2q_" + std::to_string(i)));
  }

  // Main test loop: check per-qubit prob match after each gate
  auto& pool = km.getDefaultPool();

  for (const auto& item : pool) {
    randomizeSV();
    ConstQuantumGatePtr gate = item.kernel->gate;

    /* apply gate on GPU */
    km.setLaunchConfig(svCUDA.getDevicePtr(), svCUDA.nQubits());
    km.enqueueKernelLaunch(*item.kernel);
    km.syncKernelExecution();

    // check norm
    suite.assertCloseF64(
        svCUDA.norm(), 1.0, "CUDA SV norm equals 1", GET_INFO());

    svCPU.applyGate(*llvm::dyn_cast<StandardQuantumGate>(gate.get()));

    // compare per-qubit probabilities
    for (int q : gate->qubits()) {
      suite.assertCloseF64(svCUDA.prob(q),
                           svCPU.prob(q),
                           "CUDA vs CPU probability match for qubit " +
                               std::to_string(q),
                           GET_INFO());
    }
  }
  suite.displayResult();
}

} // anonymous namespace

void test::test_cudaU2() {
  runU2qTest<2>();
  runU2qTest<6>();
  runU2qTest<12>();
}