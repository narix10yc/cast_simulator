// #include "simulation/KernelManager.h"
// #include "tests/TestKit.h"
// #include "simulation/StatevectorCPU.h"
// #include "simulation/StatevectorCUDA.h"
// #include <random>

// using namespace cast;
// using namespace cast::test;

// template<unsigned nQubits>
// static void f() {
//   test::TestSuite suite("Gate U2q (" + std::to_string(nQubits) + " qubits)");

//   CUDAKernelManager kernelMgrCUDA;
//   utils::StatevectorCPU<double> svCPU(nQubits, /* simd_s */ 0);
//   utils::StatevectorCUDA<double> svCUDA0(nQubits), svCUDA1(nQubits);

//   const auto randomizeSV = [&]() {
//     svCUDA0.randomize();
//     svCUDA1 = svCUDA0;
//     cudaMemcpy(svCPU.data(), svCUDA0.dData(), svCUDA0.sizeInBytes(),
//       cudaMemcpyDeviceToHost);
//   };

//   // Generate 2-qubit random unitary gates
//   std::vector<std::shared_ptr<QuantumGate>> gates;
//   gates.reserve(nQubits / 2);
//   for (int q = 0; q < nQubits - 1; q += 2) { // Non-overlapping pairs
//     gates.emplace_back(
//       std::make_shared<QuantumGate>(QuantumGate::RandomUnitary({q, q + 1})));
//   }

//   CUDAKernelGenConfig cudaGenConfig;
//   cudaGenConfig.matrixLoadMode = CUDAKernelGenConfig::UseMatImmValues;
//   for (size_t i = 0; i < gates.size(); i++) {
//     kernelMgrCUDA.genCUDAGate(
//       cudaGenConfig, gates[i], "gateImm_2q_" + std::to_string(i), nQubits);
//   }

//   kernelMgrCUDA.emitPTX(gates.size(), llvm::OptimizationLevel::O1, /* verbose */ 0);
//   kernelMgrCUDA.initCUJIT(gates.size(), /* verbose */ 0);

//   for (size_t i = 0; i < gates.size(); i++) {
//     randomizeSV();
//     std::stringstream ss;
//     auto& qubits = gates[i]->qubits; // {q, q+1}
//     ss << "Apply U2q at " << qubits[0] << "," << qubits[1] << ": ";
//     kernelMgrCUDA.launchCUDAKernel(
//       svCUDA0.dData(), svCUDA0.nQubits(), kernelMgrCUDA.kernels()[i]);
//     suite.assertClose(svCUDA0.norm(), 1.0,
//       ss.str() + "CUDA SV norm equals to 1", GET_INFO());

//     svCPU.applyGate(*gates[i]);
//     for (int q : qubits) {
//       suite.assertClose(svCUDA0.prob(q), svCPU.prob(q),
//         ss.str() + "CUDA and CPU SV prob match for qubit " + std::to_string(q), GET_INFO());
//     }
//   }
//   suite.displayResult();
// }

// void test::test_cudaU2() {
//   f<2>();
//   f<8>();
//   f<12>();
// }

//------------------------------------------------------------------------------
//  tests/cuda_u2q_debug_test.cpp
//------------------------------------------------------------------------------
// Same licence / copyright header as the rest of your project.
//------------------------------------------------------------------------------

#include "simulation/KernelManager.h"
#include "tests/TestKit.h"
#include "simulation/StatevectorCPU.h"
#include "simulation/StatevectorCUDA.h"

#include <iostream>   // for std::cerr diagnostics
#include <random>
#include <sstream>
#include <vector>
#include <memory>

using namespace cast;
using namespace cast::test;

namespace {

/*-------------------------------------------------------------------------*//**
 * Helper that dumps an entire state-vector to std::cerr.
 *--------------------------------------------------------------------------*/
inline void dumpStatevector(const char* tag,
                            const std::vector<std::complex<double>>& sv)
{
  std::cerr << tag << '\n';
  for (std::size_t i = 0; i < sv.size(); ++i) {
    std::cerr << "  State[" << i << "] = (" << sv[i].real()
              << ", " << sv[i].imag() << ")\n";
  }
}

/*-------------------------------------------------------------------------*//**
 * 2-qubit-gate test with full debug logging.
 *--------------------------------------------------------------------------*/
template<unsigned nQubits>
void runU2qTest()
{
  TestSuite suite("Gate U2q (" + std::to_string(nQubits) + " qubits)");

  CUDAKernelManager                kernelMgrCUDA;
  utils::StatevectorCPU<double>    svCPU  (nQubits, /*simd_s*/0);
  utils::StatevectorCUDA<double>   svCUDA0(nQubits),
                                   svCUDA1(nQubits);

  /* Random-initialisation lambda ------------------------------------------------*/
  const auto randomizeSV = [&]() {
    svCUDA0.randomize();
    svCUDA1 = svCUDA0;   // keep a spare copy if you need it
    cudaMemcpy(svCPU.data(), svCUDA0.dData(), svCUDA0.sizeInBytes(),
               cudaMemcpyDeviceToHost);
  };

  /* Build one non-overlapping 2-qubit gate per pair -----------------------------*/
  std::vector<std::shared_ptr<QuantumGate>> gates;
  gates.reserve(nQubits / 2);
  for (int q = 0; q < nQubits - 1; q += 2) {
    gates.emplace_back(std::make_shared<QuantumGate>(
        QuantumGate::RandomUnitary(std::vector<int>{q, q + 1})));
  }

  /* Generate one CUDA kernel per gate -------------------------------------------*/
  CUDAKernelGenConfig cgCfg;
  cgCfg.matrixLoadMode = CUDAKernelGenConfig::UseMatImmValues;
  for (std::size_t i = 0; i < gates.size(); ++i) {
    kernelMgrCUDA.genCUDAGate(cgCfg, gates[i],
                              "gateImm_2q_" + std::to_string(i), nQubits);
  }
  kernelMgrCUDA.emitPTX   (gates.size(), llvm::OptimizationLevel::O1, /*verbose*/0);
  kernelMgrCUDA.initCUJIT(gates.size(), /*verbose*/0);

  /* ------------------------------ main test loop ------------------------------*/
  for (std::size_t i = 0; i < gates.size(); ++i)
  {
    randomizeSV();
    auto& qubits = gates[i]->qubits;   // {q, q+1}

    /* ----- diagnostics: gate matrix ------------------------------------------*/
    std::cerr << "\n---------------------------------------------\n";
    std::cerr << "Gate " << i << " acting on qubits {"
              << qubits[0] << "," << qubits[1] << "}\n";
    std::cerr << "Expected gate matrix:\n";
    gates[i]->gateMatrix.printCMat(std::cerr) << "\n";

    /* ----- diagnostics: initial CUDA statevector -----------------------------*/
    std::vector<std::complex<double>> hostSV(1ULL << svCUDA0.nQubits());
    cudaMemcpy(hostSV.data(), svCUDA0.dData(), svCUDA0.sizeInBytes(),
               cudaMemcpyDeviceToHost);
    dumpStatevector("Initial CUDA statevector:", hostSV);

    /* ---- apply gate on GPU ---------------------------------------------------*/
    kernelMgrCUDA.launchCUDAKernel(
        svCUDA0.dData(), svCUDA0.nQubits(), kernelMgrCUDA.kernels()[i]);
    cudaDeviceSynchronize();

    /* ----- diagnostics: final CUDA statevector -------------------------------*/
    cudaMemcpy(hostSV.data(), svCUDA0.dData(), svCUDA0.sizeInBytes(),
               cudaMemcpyDeviceToHost);
    dumpStatevector("Final CUDA statevector:", hostSV);

    /* ----- check norm ---------------------------------------------------------*/
    suite.assertClose(svCUDA0.norm(), 1.0,
        "CUDA SV norm equals 1", GET_INFO());

    /* Apply same gate on CPU for cross-check -----------------------------------*/
    svCPU.applyGate(*gates[i]);

    /* ----- diagnostics: final CPU statevector --------------------------------*/
    std::cerr << "Final CPU statevector:\n";
    for (std::size_t j = 0; j < (1ULL << svCPU.nQubits()); ++j) {
      std::cerr << "  State[" << j << "] = (" << svCPU.data()[2*j]
                << ", " << svCPU.data()[2*j+1] << ")\n";
    }

    /* ----- compare per-qubit probabilities -----------------------------------*/
    for (int q : qubits) {
      double pCUDA = svCUDA0.prob(q);
      double pCPU  = svCPU.prob(q);
      std::cerr << "Qubit " << q << ": CUDA prob = " << pCUDA
                << ", CPU prob = " << pCPU << "\n";
      suite.assertClose(pCUDA, pCPU,
          "CUDA vs CPU probability match for qubit " + std::to_string(q),
          GET_INFO());
    }
  }
  suite.displayResult();
}

} // anonymous namespace

/*---------------------------------------------------------------------------*//**
 *  Entry points that the TestKit runner discovers.
 *--------------------------------------------------------------------------*/
void test::test_cudaU2()
{
  runU2qTest<2>();
  runU2qTest<6>();
  // runU2qTest<12>();
}