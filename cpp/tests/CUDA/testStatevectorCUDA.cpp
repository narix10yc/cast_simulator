#include "cast/CPU/CPUStatevector.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "tests/TestKit.h"
#include <llvm/ADT/Twine.h>

using namespace cast::test;

template <int nQubits> static bool f() {
  cast::test::TestSuite suite("CUDAStatevector CPU/GPU consistency (" +
                              std::to_string(nQubits) + " qubits)");
  cast::CUDAStatevector<float> svCudaFP32(nQubits);
  cast::CUDAStatevector<double> svCudaFP64(nQubits);

  // We have to use W0 here to allow memcpy from cuda sv to cpu sv
  cast::CPUStatevector<float> svCpuFP32(nQubits, cast::CPUSimdWidth::W0);
  cast::CPUStatevector<double> svCpuFP64(nQubits, cast::CPUSimdWidth::W0);

  svCudaFP32.initialize();
  svCudaFP64.initialize();
  suite.assertCloseFP32(svCudaFP32.norm(),
                        1.0f,
                        "FP32 initialization preserves norm",
                        GET_INFO());
  suite.assertCloseFP64(
      svCudaFP64.norm(), 1.0, "FP64 initialization preserves norm", GET_INFO());
  for (int q = 0; q < nQubits; ++q) {
    suite.assertCloseFP32(svCudaFP32.prob(q),
                          0.0f,
                          "FP32 init: qubit-" + std::to_string(q) +
                              " marginal probability is zero",
                          GET_INFO());
    suite.assertCloseFP64(svCudaFP64.prob(q),
                          0.0,
                          "FP64 init: qubit-" + std::to_string(q) +
                              " marginal probability is zero",
                          GET_INFO());
  }

  svCudaFP32.randomize();
  svCudaFP64.randomize();
  cudaMemcpy(svCpuFP32.data(),
             svCudaFP32.dData(),
             svCudaFP32.sizeInBytes(),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(svCpuFP64.data(),
             svCudaFP64.dData(),
             svCudaFP64.sizeInBytes(),
             cudaMemcpyDeviceToHost);

  suite.assertCloseFP32(
      svCudaFP32.norm(), 1.0f, "FP32 random state preserves norm", GET_INFO());
  suite.assertCloseFP64(
      svCudaFP64.norm(), 1.0, "FP64 random state preserves norm", GET_INFO());

  for (int q = 0; q < nQubits; ++q) {
    suite.assertCloseFP32(static_cast<float>(svCudaFP32.prob(q)),
                          svCpuFP32.prob(q),
                          "FP32 random state: GPU/CPU qubit-" +
                              std::to_string(q) + " probability matches",
                          GET_INFO());
    suite.assertCloseFP64(static_cast<double>(svCudaFP64.prob(q)),
                          svCpuFP64.prob(q),
                          "FP64 random state: GPU/CPU qubit-" +
                              std::to_string(q) + " probability matches",
                          GET_INFO());
  }

  return suite.displayResult();
}

bool cast::test::test_statevectorCUDA() {
  const bool ok8 = f<8>();
  const bool ok12 = f<12>();
  const bool ok16 = f<16>();
  return ok8 && ok12 && ok16;
}
