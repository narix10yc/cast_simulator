#include "simulation/KernelManager.h"
#include "tests/TestKit.h"
#include "utils/statevector.h"
#include <random>

using namespace saot;

template<unsigned simd_s, unsigned nQubits>
static void internal_U1q() {
  test::TestSuite suite(
    "Gate U1q (s=" + std::to_string(simd_s) +
    ", n=" + std::to_string(nQubits) + ")");
  utils::StatevectorAlt<double>
    sv0(nQubits, simd_s), sv1(nQubits, simd_s), sv2(nQubits, simd_s);

  const auto randomizeSV = [&sv0, &sv1, &sv2]() {
    sv0.randomize();
    sv1 = sv0;
    sv2 = sv0;
  };

  KernelManager kernelMgr;

  // generate random unitary gates
  std::vector<QuantumGate> gates;
  gates.reserve(nQubits);
  for (int q = 0; q < nQubits; q++)
    gates.emplace_back(QuantumGate::RandomUnitary(q));

  CPUKernelGenConfig cpuConfig;
  cpuConfig.simd_s = simd_s;
  cpuConfig.matrixLoadMode = CPUKernelGenConfig::UseMatImmValues;
  for (int q = 0; q < nQubits; q++)
    kernelMgr.genCPUKernel(cpuConfig, gates[q], "gateImm_" + std::to_string(q));

  cpuConfig.forceDenseKernel = true;
  cpuConfig.matrixLoadMode = CPUKernelGenConfig::StackLoadMatElems;
  for (int q = 0; q < nQubits; q++)
    kernelMgr.genCPUKernel(cpuConfig, gates[q], "gateLoad_" + std::to_string(q));

  kernelMgr.initJIT();
  for (unsigned i = 0; i < nQubits; i++) {
    randomizeSV();
    std::stringstream ss;
    ss << "Apply U1q at " << gates[i].qubits[0];
    auto immFuncName = "gateImm_" + std::to_string(i);
    auto loadFuncName = "gateLoad_" + std::to_string(i);
    kernelMgr.applyCPUKernel(sv0.data, sv0.nQubits, immFuncName);
    kernelMgr.applyCPUKernel(sv1.data, sv1.nQubits, loadFuncName);
    sv2.applyGate(gates[i]);
    suite.assertClose(sv0.norm(), 1.0, ss.str() + ": Imm Norm", GET_INFO());
    suite.assertClose(sv1.norm(), 1.0, ss.str() + ": Load Norm", GET_INFO());
    suite.assertClose(utils::fidelity(sv0, sv2), 1.0,
      ss.str() + ": Imm Fidelity", GET_INFO());
    suite.assertClose(utils::fidelity(sv1, sv2), 1.0,
      ss.str() + ": Load Fidelity", GET_INFO());
  }
  suite.displayResult();
}

template<unsigned simd_s, unsigned nQubits>
static void internal_U2q() {
  test::TestSuite suite(
    "Gate U2q (s=" + std::to_string(simd_s) +
    ", n=" + std::to_string(nQubits) + ")");
  utils::StatevectorAlt<double>
    sv0(nQubits, simd_s), sv1(nQubits, simd_s), sv2(nQubits, simd_s);

  const auto randomizeSV = [&sv0, &sv1, &sv2]() {
    sv0.randomize();
    sv1 = sv0;
    sv2 = sv0;
  };

  KernelManager kernelMgr;

  // generate random gates, set up kernel names
  std::vector<QuantumGate> gates;
  gates.reserve(nQubits);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> d(0, nQubits - 1);
  for (unsigned i = 0; i < nQubits; i++) {
    int a, b;
    a = d(gen);
    do { b = d(gen); } while (b == a);
    gates.emplace_back(QuantumGate::RandomUnitary(a, b));
  }

  CPUKernelGenConfig cpuConfig;
  cpuConfig.simd_s = simd_s;
  cpuConfig.matrixLoadMode = CPUKernelGenConfig::UseMatImmValues;
  for (int q = 0; q < nQubits; q++)
    kernelMgr.genCPUKernel(cpuConfig, gates[q], "gateImm_" + std::to_string(q));
  cpuConfig.forceDenseKernel = true;
  cpuConfig.matrixLoadMode = CPUKernelGenConfig::StackLoadMatElems;
  for (int q = 0; q < nQubits; q++)
    kernelMgr.genCPUKernel(cpuConfig, gates[q], "gateLoad_" + std::to_string(q));

  kernelMgr.initJIT();
  for (unsigned i = 0; i < nQubits; i++) {
    randomizeSV();
    int a = gates[i].qubits[0];
    int b = gates[i].qubits[1];
    std::stringstream ss;
    ss << "Apply U2q at " << a << " and " << b;
    auto immFuncName = "gateImm_" + std::to_string(i);
    auto loadFuncName = "gateLoad_" + std::to_string(i);
    kernelMgr.applyCPUKernel(sv0.data, sv0.nQubits, immFuncName);
    kernelMgr.applyCPUKernel(sv1.data, sv1.nQubits, loadFuncName);
    sv2.applyGate(gates[i]);
    suite.assertClose(sv0.norm(), 1.0, ss.str() + ": Imm Norm", GET_INFO());
    suite.assertClose(sv1.norm(), 1.0, ss.str() + ": Load Norm", GET_INFO());
    suite.assertClose(utils::fidelity(sv0, sv2), 1.0,
      ss.str() + ": Imm Fidelity", GET_INFO());
    suite.assertClose(utils::fidelity(sv1, sv2), 1.0,
      ss.str() + ": Load Fidelity", GET_INFO());
  }

  suite.displayResult();
}

void test::test_cpuU() {
  internal_U1q<1, 8>();
  internal_U1q<2, 12>();
  internal_U2q<1, 8>();
  internal_U2q<2, 8>();
}