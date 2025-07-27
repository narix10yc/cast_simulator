#include "cast/CPU/CPUStatevector.h"
#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "tests/TestKit.h"

using namespace cast;
using namespace cast::test;

namespace {

inline void dumpStatevector(const char* tag,
                            const std::vector<std::complex<double>>& sv) {
  std::cerr << tag << '\n';
  for (std::size_t i = 0; i < sv.size(); ++i) {
    std::cerr << "  State[" << i << "] = (" << sv[i].real() << ", "
              << sv[i].imag() << ")\n";
  }
}

template <unsigned nQubits> void runU2qTest() {
  TestSuite suite("Gate U2q (" + std::to_string(nQubits) + " qubits)");

  CUDAKernelManager kernelMgrCUDA;
  cast::CPUStatevector<double> svCPU(nQubits, cast::get_cpu_simd_width());
  cast::CUDAStatevector<double> svCUDA0(nQubits), svCUDA1(nQubits);

  /* Random-initialisation lambda */
  const auto randomizeSV = [&]() {
    svCUDA0.randomize();
    svCUDA1 = svCUDA0; // keep a spare copy if you need it
    cudaMemcpy(svCPU.data(),
               svCUDA0.dData(),
               svCUDA0.sizeInBytes(),
               cudaMemcpyDeviceToHost);
  };

  /* Build one non-overlapping 2-qubit gate per pair */
  std::vector<std::shared_ptr<QuantumGate>> gates;
  gates.reserve(nQubits / 2);
  for (int q = 0; q < nQubits - 1; q += 2) {
    gates.emplace_back(StandardQuantumGate::RandomUnitary({q, q + 1}));
  }

  /* Generate one CUDA kernel per gate */
  CUDAKernelGenConfig cgCfg;
  cgCfg.matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;
  for (std::size_t i = 0; i < gates.size(); ++i) {
    kernelMgrCUDA
        .genStandaloneGate(cgCfg, gates[i], "gateImm_2q_" + std::to_string(i))
        .consumeError();
  }
  kernelMgrCUDA.emitPTX(
      gates.size(), llvm::OptimizationLevel::O1, /*verbose*/ 0);
  kernelMgrCUDA.initCUJIT(gates.size(), /*verbose*/ 0);

  /* main test loop */
  for (std::size_t i = 0; i < gates.size(); ++i) {
    randomizeSV();
    auto& qubits = gates[i]->qubits(); // {q, q+1}

    /* diagnostics: gate matrix */
    // std::cerr << "\n---------------------------------------------\n";
    // std::cerr << "Gate " << i << " acting on qubits {"
    //           << qubits[0] << "," << qubits[1] << "}\n";
    // std::cerr << "Expected gate matrix:\n";
    // gates[i]->gateMatrix.printCMat(std::cerr) << "\n";

    /* diagnostics: initial CUDA statevector */
    std::vector<std::complex<double>> hostSV(1ULL << svCUDA0.nQubits());
    cudaMemcpy(hostSV.data(),
               svCUDA0.dData(),
               svCUDA0.sizeInBytes(),
               cudaMemcpyDeviceToHost);
    // dumpStatevector("Initial CUDA statevector:", hostSV);

    /* apply gate on GPU */
    kernelMgrCUDA.launchCUDAKernel(svCUDA0.dData(),
                                   svCUDA0.nQubits(),
                                   *kernelMgrCUDA.getAllStandaloneKernels()[i]);
    cudaDeviceSynchronize();

    /* diagnostics: final CUDA statevector */
    cudaMemcpy(hostSV.data(),
               svCUDA0.dData(),
               svCUDA0.sizeInBytes(),
               cudaMemcpyDeviceToHost);
    // dumpStatevector("Final CUDA statevector:", hostSV);

    /* check norm */
    suite.assertCloseF64(
        svCUDA0.norm(), 1.0, "CUDA SV norm equals 1", GET_INFO());

    /* Apply same gate on CPU for cross-check */
    svCPU.applyGate(*llvm::dyn_cast<StandardQuantumGate>(gates[i].get()));

    /* diagnostics: final CPU statevector */
    // std::cerr << "Final CPU statevector:\n";
    // for (std::size_t j = 0; j < (1ULL << svCPU.nQubits()); ++j) {
    //   std::cerr << "  State[" << j << "] = (" << svCPU.data()[2*j]
    //             << ", " << svCPU.data()[2*j+1] << ")\n";
    // }

    /* compare per-qubit probabilities */
    for (int q : qubits) {
      double pCUDA = svCUDA0.prob(q);
      double pCPU = svCPU.prob(q);
      // std::cerr << "Qubit " << q << ": CUDA prob = " << pCUDA
      //            << ", CPU prob = " << pCPU << "\n";
      suite.assertCloseF64(pCUDA,
                           pCPU,
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
  // runU2qTest<12>();
}