#include "tests/TestKit.h"
#include "cast/CPU/StatevectorCPU.h"
#include "cast/CUDA/StatevectorCUDA.h"

using namespace cast::test;

template<int nQubits>
static void f() {
  cast::test::TestSuite suite(
    "StatevectorCUDA with " + std::to_string(nQubits) + " qubits");
  utils::StatevectorCUDA<float> svCudaF32(nQubits);
  utils::StatevectorCUDA<double> svCudaF64(nQubits);
  utils::StatevectorCPU<float> svCpuF32(nQubits, /* simd_s */ 0);
  utils::StatevectorCPU<double> svCpuF64(nQubits, /* simd_s */ 0);

  svCudaF32.initialize();
  svCudaF64.initialize();
  suite.assertClose(svCudaF32.norm(), 1.0f, "initialize norm F32", GET_INFO());
  suite.assertClose(svCudaF64.norm(), 1.0, "initialize norm F64", GET_INFO());
  for (int q = 0; q < nQubits; ++q) {
    suite.assertClose(svCudaF32.prob(q), 0.0f,
      "Init SV F32: Prob of qubit " + std::to_string(q), GET_INFO());
    suite.assertClose(svCudaF64.prob(q), 0.0,
      "Init SV F64: Prob of qubit " + std::to_string(q), GET_INFO());
  }

  svCudaF32.randomize();
  svCudaF64.randomize();
  cudaMemcpy(svCpuF32.data(), svCudaF32.dData(), svCudaF32.sizeInBytes(),
    cudaMemcpyDeviceToHost);
  cudaMemcpy(svCpuF64.data(), svCudaF64.dData(), svCudaF64.sizeInBytes(),
    cudaMemcpyDeviceToHost);
  suite.assertClose(svCudaF32.norm(), 1.0f, "randomize norm F32", GET_INFO());
  suite.assertClose(svCudaF64.norm(), 1.0, "randomize norm F64", GET_INFO());

  for (int q = 0; q < nQubits; ++q) {
    suite.assertClose(svCudaF32.prob(q), svCpuF32.prob(q),
      "Rand SV F32: Prob of qubit " + std::to_string(q), GET_INFO());
    suite.assertClose(svCudaF64.prob(q), svCpuF64.prob(q),
      "Rand SV F64: Prob of qubit " + std::to_string(q), GET_INFO());
  }

  suite.displayResult();
}

void cast::test::test_statevectorCUDA() {
  f<8>();
  f<12>();
  f<16>();
}
