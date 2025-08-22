#include "cast/CPU/CPUStatevector.h"
#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "tests/TestKit.h"

#include <algorithm>
#include <complex>
#include <cuda_runtime.h>
#include "cast/CUDA/CUDAPermute.h"

using namespace cast;
using namespace cast::test;

static inline void set_layout_LSB(cast::BitLayout& layout,
                                  const std::vector<int>& Q,
                                  int nSys) {
  const int k = (int)Q.size();
  std::vector<char> isTarget(nSys, 0);
  for (int b = 0; b < k; ++b) isTarget[Q[b]] = 1;

  // non-targets in ascending old physical order
  std::vector<std::pair<int,int>> others;
  others.reserve(std::max(0, nSys - k));
  for (int l = 0; l < nSys; ++l)
    if (!isTarget[l]) others.emplace_back(layout.phys_of_log[l], l);
  std::sort(others.begin(), others.end());

  for (int b = 0; b < k; ++b) layout.phys_of_log[Q[b]] = b;
  for (int i = 0; i < (int)others.size(); ++i)
    layout.phys_of_log[others[i].second] = k + i;

  for (int p = 0; p < nSys; ++p) layout.log_of_phys[p] = -1;
  for (int l = 0; l < nSys; ++l) layout.log_of_phys[layout.phys_of_log[l]] = l;
}

template<typename ScalarType>
static inline void permute_to_LSBs_if_needed(
    ScalarType*& dCurrent,
    ScalarType*  dScratch,
    int          nSys,
    const std::vector<int>& logicalQubits,
    cast::BitLayout& layout,
    cudaStream_t stream = 0)
{
  const int k = (int)logicalQubits.size();
  if (k == 0) return;

  bool lsbOK = true;
  for (int b = 0; b < k; ++b)
    if (layout.phys_of_log[logicalQubits[b]] != b) { lsbOK = false; break; }
  if (lsbOK) return;

  uint64_t maskLow = 0;
  for (int b = 0; b < k; ++b)
    maskLow |= (1ull << layout.phys_of_log[logicalQubits[b]]);

  // Reorder interleaved (re,im) pairs so that 'logicalQubits' occupy the LSBs
  cast_permute_lowbits<ScalarType>(dCurrent, dScratch, nSys, maskLow, k, stream);
  std::swap(dCurrent, dScratch);

  set_layout_LSB(layout, logicalQubits, nSys);
}

/* compute Pr(q=1) from a host copy of the SV, given the *physical* bit index */
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
  cudaGenConfig.precision = Precision::F64;
  cudaGenConfig.matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;
  cudaGenConfig.assumeContiguousTargets = true;  // kernels expect targets on LSBs

  for (size_t i = 0; i < gates.size(); i++) {
    auto funcName = "gateImm_" + std::to_string(targetQubits.size()) + "q_" +
                    std::to_string(i);
    kernelMgrCUDA.genStandaloneGate(cudaGenConfig, gates[i], funcName)
        .consumeError();
  }

  kernelMgrCUDA.emitPTX(
      gates.size(), llvm::OptimizationLevel::O1, /* verbose */ 0);
  kernelMgrCUDA.initCUJIT(gates.size(), /* verbose */ 0);

  // layout + scratch used by permutation
  BitLayout layout; layout.init(nQubits);
  double* dScratch = nullptr;
  {
    const uint64_t nComplex = 1ull << nQubits;             // number of complex amplitudes
    const size_t   nScalars = size_t(2) * nComplex;        // interleaved re/im
    cudaMalloc(&dScratch, nScalars * sizeof(double));
  }

  for (size_t i = 0; i < gates.size(); i++) {
    randomizeSV();
    layout.init(nQubits);  // reset mapping for each fresh randomized state

    // Reset dCurrent to the actual SV buffer for this iteration
    double* dCurrent = reinterpret_cast<double*>(svCUDA0.dData());

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
               dCurrent,                        // <-- read from *current* buffer
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
    for (int i2 = 0; i2 < (1 << (int)targetQubits.size()); ++i2) {
      uint64_t delta = 0;
      for (size_t b = 0; b < targetQubits.size(); ++b) {
        if (i2 & (1 << (int)b)) {
          delta |= (1ULL << targetQubits[b]);
        }
      }
      // std::cerr << "i=" << i2 << ", delta=" << delta << "\n";
      delta |= (1ULL << 1);
      // std::cerr << "i=" << i2 << ", delta (qubit 1=1)=" << delta << "\n";
    }

    // ====== permute so that target qubits occupy LSBs for this kernel ======
    {
      const CUDAKernelInfo* KInfo =
          kernelMgrCUDA.getAllStandaloneKernels()[i].get();   // <-- .get() !
      const auto& Q = KInfo->gate->qubits();                  // logical indices of the gate
      permute_to_LSBs_if_needed<double>(dCurrent, dScratch, svCUDA0.nQubits(),
                                        Q, layout, /*stream*/0);
    }

    // Launch on the *current* buffer (which may now be the scratch)
    kernelMgrCUDA.launchCUDAKernel(static_cast<void*>(dCurrent),
                                   svCUDA0.nQubits(),
                                   *kernelMgrCUDA.getAllStandaloneKernels()[i],
                                   1 << targetQubits.size());
    cudaDeviceSynchronize();

    // Print final CUDA statevector
    // std::cerr << "Final CUDA Statevector:\n";
    cudaMemcpy(svCUDA0_data.data(),
               dCurrent,                        // <-- read from *current* buffer
               svCUDA0.sizeInBytes(),
               cudaMemcpyDeviceToHost);
    // for (size_t j = 0; j < svCUDA0_data.size(); j++) {
    //     std::cerr << "State[" << j << "] = (" << svCUDA0_data[j].real() << ",
    //     " << svCUDA0_data[j].imag() << ")\n";
    // }

    // Compute norm from the host copy (the SV object still points to the original buffer)
    double hostNorm = 0.0;
    for (const auto& c : svCUDA0_data) hostNorm += std::norm(c);
    suite.assertCloseF64(
        hostNorm, 1.0, ss.str() + "CUDA SV norm equals to 1", GET_INFO());

    svCPU.applyGate(*llvm::dyn_cast<StandardQuantumGate>(gates[i].get()));
    // std::cerr << "Final CPU Statevector:\n";
    // for (size_t j = 0; j < (1 << svCPU.nQubits()); j++) {
    //     std::cerr << "State[" << j << "] = (" << svCPU.data()[2*j] << ", " <<
    //     svCPU.data()[2*j+1] << ")\n";
    // }

    for (int q : targetQubits) {
      // CUDA prob: interpret logical q using the *current* layout (physical bit position)
      const int phys = layout.phys_of_log[q];
      double cudaProb = prob_from_host_sv(svCUDA0_data, svCUDA0.nQubits(), phys);

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

  if (dScratch) cudaFree(dScratch);
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