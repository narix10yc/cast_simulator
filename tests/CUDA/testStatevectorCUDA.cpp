#include "cast/CPU/CPUStatevector.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "tests/TestKit.h"
#include "llvm/ADT/Twine.h"

using namespace cast::test;

template <int nQubits> static void f() {
  cast::test::TestSuite suite("CUDAStatevector with " +
                              std::to_string(nQubits) + " qubits");
  cast::CUDAStatevector<float> svCudaFP32(nQubits);
  cast::CUDAStatevector<double> svCudaFP64(nQubits);

  // We have to use W0 here to allow memcpy from cuda sv to cpu sv
  cast::CPUStatevector<float> svCpuFP32(nQubits, cast::CPUSimdWidth::W0);
  cast::CPUStatevector<double> svCpuFP64(nQubits, cast::CPUSimdWidth::W0);

  svCudaFP32.initialize();
  svCudaFP64.initialize();
  suite.assertCloseFP32(
      svCudaFP32.norm(), 1.0f, "initialize norm FP32", GET_INFO());
  suite.assertCloseFP64(
      svCudaFP64.norm(), 1.0, "initialize norm FP64", GET_INFO());
  for (int q = 0; q < nQubits; ++q) {
    suite.assertCloseFP32(svCudaFP32.prob(q),
                         0.0f,
                         "Init SV FP32: Prob of qubit " + std::to_string(q),
                         GET_INFO());
    suite.assertCloseFP64(svCudaFP64.prob(q),
                         0.0,
                         "Init SV FP64: Prob of qubit " + std::to_string(q),
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

  suite.assertCloseFP32(svCudaFP32.norm(), 1.0f, "Rand SV FP32: Norm", GET_INFO());
  suite.assertCloseFP64(svCudaFP64.norm(), 1.0, "Rand SV FP64: Norm", GET_INFO());

  for (int q = 0; q < nQubits; ++q) {
    suite.assertCloseFP32(static_cast<float>(svCudaFP32.prob(q)),
                         svCpuFP32.prob(q),
                         "Rand SV FP32: Prob of qubit " + std::to_string(q),
                         GET_INFO());
    suite.assertCloseFP64(static_cast<double>(svCudaFP64.prob(q)),
                         svCpuFP64.prob(q),
                         "Rand SV FP64: Prob of qubit " + std::to_string(q),
                         GET_INFO());
  }

  suite.displayResult();
}

void cast::test::test_statevectorCUDA() {
  f<8>();
  f<12>();
  f<16>();
}
