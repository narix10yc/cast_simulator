#include "cast/CPU/CPUStatevector.h"
#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "tests/TestKit.h"

#include <memory>

using namespace cast;
using namespace cast::test;

template <int nQubits> static void f() {
  test::TestSuite suite("Gate U1q (" + std::to_string(nQubits) + " qubits)");

  // we have to use W0 here to allow memcpy from cuda sv to cpu sv
  cast::CPUStatevector<double> svCPU(nQubits, cast::CPUSimdWidth::W0);
  auto svCUDA = std::make_unique<cast::CUDAStatevector<double>>(nQubits);
  auto* svCUDAPtr = svCUDA.get();

  // use 2 worker threads
  CUDAKernelManager km(2);
  km.attachStatevector(std::move(svCUDA));

  const auto randomizeSV = [&]() {
    svCUDAPtr->randomize();
    cudaMemcpy(svCPU.data(),
               svCUDAPtr->dData(),
               svCUDAPtr->sizeInBytes(),
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

  std::vector<CUDAKernelHandler> handlers;
  handlers.reserve(nQubits);
  for (int q = 0; q < nQubits; q++) {
    auto eKernel = km.genGate(cfg, gates[q], "gateImm_" + std::to_string(q));
    if (!eKernel) {
      suite.check(eKernel.takeError(), "genGate", GET_INFO());
      return;
    }
    handlers.push_back(*eKernel);
  }
  CHECK(suite, km.syncCompilation());

  for (int q = 0; q < nQubits; q++) {
    std::stringstream ss;
    assert(q == gates[q]->qubits()[0]);
    ss << "Apply U1q at " << q << ": ";

    randomizeSV();
    suite.assertCloseFP64(svCUDAPtr->prob(q),
                          svCPU.prob(q),
                          ss.str() + "Prob match before applying gate",
                          GET_INFO());

    // Apply CPU gate
    svCPU.applyGate(*llvm::dyn_cast<StandardQuantumGate>(gates[q].get()));

    // Apply CUDA gate
    auto eTask = km.enqueueKernelExecution(handlers[q]);
    if (!eTask) {
      suite.check(eTask.takeError(), "enqueueKernelExecution", GET_INFO());
      return;
    }
    CHECK(suite, km.syncKernelExecution());

    suite.assertCloseFP64(
        svCUDAPtr->norm(),
        1.0,
        ss.str() + "CUDA SV norm equals to 1",
        GET_INFO());
    suite.assertCloseFP64(svCUDAPtr->prob(q),
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
