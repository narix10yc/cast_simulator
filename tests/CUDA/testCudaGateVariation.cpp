#include "cast/CPU/CPUStatevector.h"
#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "tests/TestKit.h"

using namespace cast;
using namespace cast::test;

template <unsigned nQubits>
static void f(const std::vector<int>& targetQubits) {
  test::TestSuite suite("Gate U" + std::to_string(targetQubits.size()) + "q (" +
                        std::to_string(nQubits) + " qubits)");

  CUDAKernelManager kernelMgrCUDA;
  cast::CPUStatevector<double> svCPU(nQubits, cast::CPUSimdWidth::W0);
  cast::CUDAStatevector<double> svCUDA0(nQubits), svCUDA1(nQubits);

  const auto randomizeSV = [&]() {
    svCUDA0.randomize();
    svCUDA1 = svCUDA0;
    cudaMemcpy(svCPU.data(),
               svCUDA0.dData(),
               svCUDA0.sizeInBytes(),
               cudaMemcpyDeviceToHost);
  };

  // Generate k-qubit random unitary gate
  std::vector<QuantumGatePtr> gates;
  gates.emplace_back(StandardQuantumGate::RandomUnitary(targetQubits));

  CUDAKernelGenConfig cudaGenConfig;
  cudaGenConfig.matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;
  for (size_t i = 0; i < gates.size(); i++) {
    auto funcName = "gateImm_" + std::to_string(targetQubits.size()) + "q_" +
                    std::to_string(i);
    kernelMgrCUDA.genStandaloneGate(cudaGenConfig, gates[i], funcName)
        .consumeError();
  }

  kernelMgrCUDA.emitPTX(
      gates.size(), llvm::OptimizationLevel::O1, /* verbose */ 0);
  kernelMgrCUDA.initCUJIT(gates.size(), /* verbose */ 0);

  for (size_t i = 0; i < gates.size(); i++) {
    randomizeSV();
    std::stringstream ss;
    // ss << "Apply U" << targetQubits.size() << "q at ";
    // for (size_t j = 0; j < targetQubits.size(); ++j) {
    //   ss << targetQubits[j];
    //   if (j < targetQubits.size() - 1) ss << ",";
    // }
    // ss << ": ";

    // Log expected gate matrix
    // std::cerr << "Expected Gate Matrix for gate " << i << ":\n";
    // gates[i]->gateMatrix.printCMat(std::cerr) << "\n";

    // std::cerr << "Initial CUDA Statevector:\n";
    std::vector<std::complex<double>> svCUDA0_data(1 << svCUDA0.nQubits());
    cudaMemcpy(svCUDA0_data.data(),
               svCUDA0.dData(),
               svCUDA0.sizeInBytes(),
               cudaMemcpyDeviceToHost);
    // for (size_t j = 0; j < svCUDA0_data.size(); j++) {
    //     std::cerr << "State[" << j << "] = (" << svCUDA0_data[j].real() << ",
    //     " << svCUDA0_data[j].imag() << ")\n";
    // }

    // std::cerr << "Expected statevector indices for qubits {";
    // for (size_t j = 0; j < targetQubits.size(); ++j) {
    //     std::cerr << targetQubits[j];
    //     if (j < targetQubits.size() - 1) std::cerr << ", ";
    // }
    // std::cerr << "}:\n";
    for (int i = 0; i < (1 << targetQubits.size()); ++i) {
      uint64_t delta = 0;
      for (size_t b = 0; b < targetQubits.size(); ++b) {
        if (i & (1 << b)) {
          delta |= (1ULL << targetQubits[b]);
        }
      }
      // std::cerr << "i=" << i << ", delta=" << delta << "\n";
      delta |= (1ULL << 1);
      // std::cerr << "i=" << i << ", delta (qubit 1=1)=" << delta << "\n";
    }

    kernelMgrCUDA.launchCUDAKernel(svCUDA0.dData(),
                                   svCUDA0.nQubits(),
                                   *kernelMgrCUDA.getAllStandaloneKernels()[i],
                                   1 << targetQubits.size());
    cudaDeviceSynchronize();

    // Print final CUDA statevector
    // std::cerr << "Final CUDA Statevector:\n";
    cudaMemcpy(svCUDA0_data.data(),
               svCUDA0.dData(),
               svCUDA0.sizeInBytes(),
               cudaMemcpyDeviceToHost);
    // for (size_t j = 0; j < svCUDA0_data.size(); j++) {
    //     std::cerr << "State[" << j << "] = (" << svCUDA0_data[j].real() << ",
    //     " << svCUDA0_data[j].imag() << ")\n";
    // }

    suite.assertCloseF64(
        svCUDA0.norm(), 1.0, ss.str() + "CUDA SV norm equals to 1", GET_INFO());

    svCPU.applyGate(*llvm::dyn_cast<StandardQuantumGate>(gates[i].get()));
    // std::cerr << "Final CPU Statevector:\n";
    // for (size_t j = 0; j < (1 << svCPU.nQubits()); j++) {
    //     std::cerr << "State[" << j << "] = (" << svCPU.data()[2*j] << ", " <<
    //     svCPU.data()[2*j+1] << ")\n";
    // }

    for (int q : targetQubits) {
      double cudaProb = svCUDA0.prob(q);
      double cpuProb = svCPU.prob(q);
      // std::cerr << "Qubit " << q << ": CUDA prob=" << cudaProb << ", CPU
      // prob=" << cpuProb << "\n";
      suite.assertCloseF64(cudaProb,
                           cpuProb,
                           ss.str() + "CUDA and CPU SV prob match for qubit " +
                               std::to_string(q),
                           GET_INFO());
    }
  }
  suite.displayResult();
}

void test::test_cuda_gate_var() {
  // Test 1: 2-qubit system, 2-qubit gate on contiguous qubits {0, 1}
  f<2>({0, 1});

  // Test 2: 3-qubit system, 3-qubit gate on contiguous qubits {0, 1, 2}
  f<3>({0, 1, 2});

  // Test 3: 3-qubit system, 2-qubit gate on contiguous qubits {0, 1}
  f<3>({0, 1});

  // Test 4: 3-qubit system, 2-qubit gate on non-contiguous qubits {0, 2}
  f<3>({0, 2});

  // Test 5: 6-qubit system, 3-qubit gate on non-contiguous qubits {2, 4, 5}
  f<6>({2, 4, 5});

  // Test 6: 10-qubit system, 4-qubit gate on non-contiguous qubits {0, 2, 3, 6}
  f<10>({0, 2, 3, 6});
}