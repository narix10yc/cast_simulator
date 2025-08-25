#include "cast/CPU/CPUStatevector.h"
#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "tests/TestKit.h"

#include <algorithm>
#include <complex>
#include <cuda_runtime.h>

using namespace cast;
using namespace cast::test;

static double prob_from_host_sv(const std::vector<std::complex<double>>& svHost,
                                int nQubits, int physBit) {
  const size_t N = svHost.size();
  double sum = 0.0;
  for (size_t idx = 0; idx < N; ++idx) {
    if ((idx >> physBit) & 1ull) sum += std::norm(svHost[idx]);
  }
  return sum;
}

template <unsigned nQubits>
static void f(const std::vector<int>& targetQubits) {
  test::TestSuite suite("Gate U" + std::to_string(targetQubits.size()) + "q (" +
                        std::to_string(nQubits) + " qubits)");

  CUDAKernelManager kernelMgrCUDA;
  cast::CPUStatevector<double>  svCPU(nQubits, cast::CPUSimdWidth::W0);
  cast::CUDAStatevector<double> svCUDA0(nQubits), svCUDA1(nQubits);

  const auto randomizeSV = [&]() {
    svCUDA0.randomize();
    svCUDA1 = svCUDA0;
    cudaMemcpy(svCPU.data(),
               svCUDA0.dData(),
               svCUDA0.sizeInBytes(),
               cudaMemcpyDeviceToHost);
  };

  // Generate k‑qubit random unitary gate
  std::vector<QuantumGatePtr> gates;
  gates.emplace_back(StandardQuantumGate::RandomUnitary(targetQubits));

  CUDAKernelGenConfig cudaGenConfig;
  cudaGenConfig.precision = Precision::F64;
  cudaGenConfig.matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;
  cudaGenConfig.assumeContiguousTargets = false;  // ← key change

  for (size_t i = 0; i < gates.size(); i++) {
    auto funcName = "gateImm_" + std::to_string(targetQubits.size()) + "q_" +
                    std::to_string(i);
    kernelMgrCUDA.genStandaloneGate(cudaGenConfig, gates[i], funcName)
        .consumeError();
  }

  kernelMgrCUDA.emitPTX(gates.size(), llvm::OptimizationLevel::O1, 0);
  kernelMgrCUDA.initCUJIT(gates.size(), 0);

  for (size_t i = 0; i < gates.size(); i++) {
    randomizeSV();

    double* dCurrent = reinterpret_cast<double*>(svCUDA0.dData());
    std::vector<std::complex<double>> svCUDA0_data(1 << svCUDA0.nQubits());
    cudaMemcpy(svCUDA0_data.data(),
               dCurrent,
               svCUDA0.sizeInBytes(),
               cudaMemcpyDeviceToHost);

    const unsigned k = static_cast<unsigned>(targetQubits.size());
    const uint32_t combos = 1u << (nQubits - k);

    kernelMgrCUDA.launchCUDAKernel(static_cast<void*>(dCurrent),
                                   svCUDA0.nQubits(),
                                   *kernelMgrCUDA.getAllStandaloneKernels()[i],
                                   combos);
    cudaDeviceSynchronize();

    cudaMemcpy(svCUDA0_data.data(),
               dCurrent,
               svCUDA0.sizeInBytes(),
               cudaMemcpyDeviceToHost);

    double hostNorm = 0.0;
    for (const auto& c : svCUDA0_data) hostNorm += std::norm(c);
    suite.assertCloseF64(hostNorm, 1.0,
                         "CUDA SV norm equals to 1", GET_INFO());

    svCPU.applyGate(*llvm::dyn_cast<StandardQuantumGate>(gates[i].get()));
    for (int q : targetQubits) {
      double cudaProb = prob_from_host_sv(svCUDA0_data, svCUDA0.nQubits(), /*physBit=*/q);
      double cpuProb  = svCPU.prob(q);
      suite.assertCloseF64(cudaProb, cpuProb,
                           "CUDA and CPU SV prob match for qubit " + std::to_string(q),
                           GET_INFO());
    }
  }

  suite.displayResult();
}

// void test::test_cuda_gate_var() {
//   // Test 1: 2-qubit system, 2-qubit gate on contiguous qubits {0, 1}
//   f<2>({0, 1});

//   // Test 2: 3-qubit system, 3-qubit gate on contiguous qubits {0, 1, 2}
//   f<3>({0, 1, 2});

//   // Test 3: 3-qubit system, 2-qubit gate on contiguous qubits {0, 1}
//   f<3>({0, 1});

//   // Test 4: 3-qubit system, 2-qubit gate on non-contiguous qubits {0, 2}
//   f<3>({0, 2});

//   // Test 5: 6-qubit system, 3-qubit gate on non-contiguous qubits {2, 4, 5}
//   f<6>({2, 4, 5});

//   // Test 6: 10-qubit system, 4-qubit gate on non-contiguous qubits {0, 2, 3, 6}
//   f<10>({0, 2, 3, 6});
// }
