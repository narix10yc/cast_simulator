#include "cast/CPU/CPUStatevector.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "tests/TestKit.h"

using namespace cast::test;

template <int nQubits> static void f() {
  cast::test::TestSuite suite("CUDAStatevector with " +
                              std::to_string(nQubits) + " qubits");
  cast::CUDAStatevector<float> svCudaF32(nQubits);
  cast::CUDAStatevector<double> svCudaF64(nQubits);
  cast::CPUStatevector<float> svCpuF32(nQubits, cast::get_cpu_simd_width());
  cast::CPUStatevector<double> svCpuF64(nQubits, cast::get_cpu_simd_width());

  svCudaF32.initialize();
  svCudaF64.initialize();
  suite.assertCloseF32(
      svCudaF32.norm(), 1.0f, "initialize norm F32", GET_INFO());
  suite.assertCloseF64(
      svCudaF64.norm(), 1.0, "initialize norm F64", GET_INFO());
  for (int q = 0; q < nQubits; ++q) {
    suite.assertCloseF32(svCudaF32.prob(q),
                         0.0f,
                         "Init SV F32: Prob of qubit " + std::to_string(q),
                         GET_INFO());
    suite.assertCloseF64(svCudaF64.prob(q),
                         0.0,
                         "Init SV F64: Prob of qubit " + std::to_string(q),
                         GET_INFO());
  }

  svCudaF32.randomize();
  svCudaF64.randomize();
  cudaMemcpy(svCpuF32.data(),
             svCudaF32.dData(),
             svCudaF32.sizeInBytes(),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(svCpuF64.data(),
             svCudaF64.dData(),
             svCudaF64.sizeInBytes(),
             cudaMemcpyDeviceToHost);
  suite.assertCloseF32(
      svCudaF32.norm(), 1.0f, "randomize norm F32", GET_INFO());
  suite.assertCloseF64(svCudaF64.norm(), 1.0, "randomize norm F64", GET_INFO());

  for (int q = 0; q < nQubits; ++q) {
    suite.assertCloseF32(static_cast<float>(svCudaF32.prob(q)),
                         svCpuF32.prob(q),
                         "Rand SV F32: Prob of qubit " + std::to_string(q),
                         GET_INFO());
    suite.assertCloseF64(static_cast<double>(svCudaF64.prob(q)),
                         svCpuF64.prob(q),
                         "Rand SV F64: Prob of qubit " + std::to_string(q),
                         GET_INFO());
  }

  suite.displayResult();
}

void cast::test::test_statevectorCUDA() {
  f<8>();
  f<12>();
  f<16>();
}
