#include "cast/CPU/CPUStatevector.h"
#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "tests/TestKit.h"

#include <memory>

using namespace cast;
using namespace cast::test;

template <int nQubits> static bool f() {
  test::TestSuite suite("CUDA U1q kernel parity (" + std::to_string(nQubits) +
                        " qubits)");

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

  std::vector<CudaKernel*> kernels;
  kernels.reserve(nQubits);
  for (int q = 0; q < nQubits; q++) {
    auto eKernel = km.genGate(cfg, gates[q], "gateImm_" + std::to_string(q));
    if (!eKernel) {
      suite.check(eKernel.takeError(), "genGate", GET_INFO());
      return suite.displayResult();
    }
    kernels.push_back(*eKernel);
  }
  CHECK(suite, km.syncCompilation());

  for (int q = 0; q < nQubits; q++) {
    std::stringstream ss;
    assert(q == gates[q]->qubits()[0]);
    ss << "Apply random U1q on qubit " << q << ": ";

    randomizeSV();
    suite.assertCloseFP64(svCUDAPtr->prob(q),
                          svCPU.prob(q),
                          ss.str() +
                              "GPU/CPU probabilities match before kernel",
                          GET_INFO());

    // Apply CPU gate
    svCPU.applyGate(*llvm::dyn_cast<StandardQuantumGate>(gates[q].get()));

    // Apply CUDA gate
    auto eTask = km.enqueueKernelExecution(kernels[q]);
    if (!eTask) {
      suite.check(eTask.takeError(), "enqueueKernelExecution", GET_INFO());
      return suite.displayResult();
    }
    CHECK(suite, km.syncKernelExecution());

    suite.assertCloseFP64(svCUDAPtr->norm(),
                          1.0,
                          ss.str() + "CUDA state preserves norm",
                          GET_INFO());
    suite.assertCloseFP64(svCUDAPtr->prob(q),
                          svCPU.prob(q),
                          ss.str() + "GPU kernel result matches CPU reference",
                          GET_INFO());
  }
  return suite.displayResult();
}

bool test::test_cudaU() {
  const bool ok4 = f<4>();
  const bool ok8 = f<8>();
  const bool ok12 = f<12>();
  return ok4 && ok8 && ok12;
}
