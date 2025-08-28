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

  // use 2 worker threads
  CUDAKernelManager km(2);
  // we have to use W0 here to allow direct memcpy between host and device
  cast::CPUStatevector<double> svCPU(nQubits, cast::CPUSimdWidth::W0);
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
  std::vector<QuantumGatePtr> gates;
  gates.reserve(nQubits / 2);
  for (int q = 0; q < nQubits - 1; q += 2) {
    gates.emplace_back(StandardQuantumGate::RandomUnitary({q, q + 1}));
  }

  /* Generate one CUDA kernel per gate */
  CUDAKernelGenConfig cgCfg;
  cgCfg.matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;
  for (std::size_t i = 0; i < gates.size(); ++i) {
    km.genStandaloneGate(cgCfg, gates[i], "gateImm_2q_" + std::to_string(i))
        .consumeError();
  }
  km.compileLLVMIRToPTX(1, /*verbose*/ 0);
  km.compilePTXToCubin(1, /*verbose*/ 0);

  /* main test loop */
  for (std::size_t i = 0; i < gates.size(); ++i) {
    randomizeSV();
    auto& qubits = gates[i]->qubits(); // {q, q+1}

    std::vector<std::complex<double>> hostSV(1ULL << svCUDA0.nQubits());
    cudaMemcpy(hostSV.data(),
               svCUDA0.dData(),
               svCUDA0.sizeInBytes(),
               cudaMemcpyDeviceToHost);
    // dumpStatevector("Initial CUDA statevector:", hostSV);

    /* apply gate on GPU */
    km.launchCUDAKernel(
        svCUDA0.dData(), svCUDA0.nQubits(), *km.getAllStandaloneKernels()[i]);
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

    // compare per-qubit probabilities
    for (int q : qubits) {
      suite.assertCloseF64(svCUDA0.prob(q),
                           svCPU.prob(q),
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