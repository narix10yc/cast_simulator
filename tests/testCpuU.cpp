#include "cast/CPU/CPUKernelManager.h"
#include "cast/CPU/CPUStatevector.h"
#include "tests/TestKit.h"
#include <random>

using namespace cast;

template <CPUSimdWidth SimdWidth, unsigned nQubits> static void internal_U1q() {
  test::TestSuite suite("Gate U1q (s=" + std::to_string(SimdWidth) +
                        ", n=" + std::to_string(nQubits) + ")");
  cast::CPUStatevector<double> sv0(nQubits, SimdWidth), sv1(nQubits, SimdWidth),
      sv2(nQubits, SimdWidth);

  const auto randomizeSV = [&sv0, &sv1, &sv2]() {
    sv0.randomize();
    sv1 = sv0;
    sv2 = sv0;
  };

  CPUKernelManager kernelMgr;

  // generate random unitary gates
  std::vector<StandardQuantumGatePtr> gates;
  gates.reserve(nQubits);
  for (int q = 0; q < nQubits; q++) {
    gates.emplace_back(StandardQuantumGate::RandomUnitary({q}));
  }

  CPUKernelGenConfig cpuConfig(SimdWidth, Precision::F64);
  cpuConfig.matrixLoadMode = CPUMatrixLoadMode::UseMatImmValues;
  for (int q = 0; q < nQubits; q++) {
    kernelMgr
        .genStandaloneGate(cpuConfig, gates[q], "gateImm_" + std::to_string(q))
        .consumeError(); // ignore possible errors
  }

  cpuConfig.zeroTol = 0.0;
  cpuConfig.oneTol = 0.0;
  cpuConfig.matrixLoadMode = CPUMatrixLoadMode::StackLoadMatElems;
  for (int q = 0; q < nQubits; q++) {
    kernelMgr
        .genStandaloneGate(cpuConfig, gates[q], "gateLoad_" + std::to_string(q))
        .consumeError(); // ignore possible errors
  }

  kernelMgr.initJIT().consumeError(); // ignore possible errors
  for (unsigned i = 0; i < nQubits; i++) {
    randomizeSV();
    std::stringstream ss;
    ss << "Apply U1q at " << gates[i]->qubits()[0];
    auto immFuncName = "gateImm_" + std::to_string(i);
    auto loadFuncName = "gateLoad_" + std::to_string(i);
    const auto* immKernel = kernelMgr.getKernelByName(immFuncName);
    const auto* loadKernel = kernelMgr.getKernelByName(loadFuncName);
    assert(immKernel);
    assert(loadKernel);
    kernelMgr.applyCPUKernel(sv0.data(), sv0.nQubits(), *immKernel)
        .consumeError();
    kernelMgr.applyCPUKernel(sv1.data(), sv1.nQubits(), *loadKernel)
        .consumeError();
    sv2.applyGate(*gates[i]);
    suite.assertClose(sv0.norm(), 1.0, ss.str() + ": Imm Norm", GET_INFO());
    suite.assertClose(sv1.norm(), 1.0, ss.str() + ": Load Norm", GET_INFO());
    suite.assertClose(
        cast::fidelity(sv0, sv2), 1.0, ss.str() + ": Imm Fidelity", GET_INFO());
    suite.assertClose(cast::fidelity(sv1, sv2),
                      1.0,
                      ss.str() + ": Load Fidelity",
                      GET_INFO());
  }
  suite.displayResult();
}

template <CPUSimdWidth SimdWidth, unsigned nQubits> static void internal_U2q() {
  test::TestSuite suite("Gate U2q (s=" + std::to_string(SimdWidth) +
                        ", n=" + std::to_string(nQubits) + ")");
  cast::CPUStatevector<double> sv0(nQubits, SimdWidth), sv1(nQubits, SimdWidth),
      sv2(nQubits, SimdWidth);

  const auto randomizeSV = [&sv0, &sv1, &sv2]() {
    sv0.randomize();
    sv1 = sv0;
    sv2 = sv0;
  };

  CPUKernelManager kernelMgr;
  // generate random gates, set up kernel names
  std::vector<StandardQuantumGatePtr> gates;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> d(0, nQubits - 1);
  for (unsigned i = 0; i < nQubits; i++) {
    int a, b;
    a = d(gen);
    do {
      b = d(gen);
    } while (b == a);
    gates.emplace_back(StandardQuantumGate::RandomUnitary({a, b}));
  }

  CPUKernelGenConfig cpuConfig(SimdWidth, Precision::F64);
  cpuConfig.matrixLoadMode = CPUMatrixLoadMode::UseMatImmValues;
  for (int q = 0; q < nQubits; q++) {
    kernelMgr
        .genStandaloneGate(cpuConfig, gates[q], "gateImm_" + std::to_string(q))
        .consumeError(); // ignore possible errors
  }
  cpuConfig.zeroTol = 0.0;
  cpuConfig.oneTol = 0.0;
  cpuConfig.matrixLoadMode = CPUMatrixLoadMode::StackLoadMatElems;
  for (int q = 0; q < nQubits; q++) {
    kernelMgr
        .genStandaloneGate(cpuConfig, gates[q], "gateLoad_" + std::to_string(q))
        .consumeError(); // ignore possible errors
  }

  kernelMgr.initJIT().consumeError(); // ignore possible errors
  for (unsigned i = 0; i < nQubits; i++) {
    randomizeSV();
    int a = gates[i]->qubits()[0];
    int b = gates[i]->qubits()[1];
    std::stringstream ss;
    ss << "Apply U2q at " << a << " and " << b;
    auto immFuncName = "gateImm_" + std::to_string(i);
    auto loadFuncName = "gateLoad_" + std::to_string(i);
    const auto* immKernel = kernelMgr.getKernelByName(immFuncName);
    const auto* loadKernel = kernelMgr.getKernelByName(loadFuncName);
    assert(immKernel);
    assert(loadKernel);
    kernelMgr.applyCPUKernel(sv0.data(), sv0.nQubits(), *immKernel)
        .consumeError();
    kernelMgr.applyCPUKernel(sv1.data(), sv1.nQubits(), *loadKernel)
        .consumeError();
    sv2.applyGate(*gates[i]);
    suite.assertClose(sv0.norm(), 1.0, ss.str() + ": Imm Norm", GET_INFO());
    suite.assertClose(sv1.norm(), 1.0, ss.str() + ": Load Norm", GET_INFO());
    suite.assertClose(
        cast::fidelity(sv0, sv2), 1.0, ss.str() + ": Imm Fidelity", GET_INFO());
    suite.assertClose(cast::fidelity(sv1, sv2),
                      1.0,
                      ss.str() + ": Load Fidelity",
                      GET_INFO());
  }

  suite.displayResult();
}

void test::test_cpuU() {
  internal_U1q<W128, 8>();
  internal_U1q<W256, 12>();
  internal_U2q<W128, 8>();
  internal_U2q<W256, 8>();
}