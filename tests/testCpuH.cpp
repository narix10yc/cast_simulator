#include "cast/CPU/CPUKernelManager.h"
#include "cast/CPU/CPUStatevector.h"
#include "tests/TestKit.h"

using namespace cast;
using namespace utils;

static QuantumGatePtr getH(int q) {
  return StandardQuantumGate::Create(ScalarGateMatrix::H(), nullptr, {q});
}

template<unsigned simd_s>
static void f() {
  test::TestSuite suite("Gate H (s = " + std::to_string(simd_s) + ")");

  CPUKernelManager cpuKernelMgr;

  CPUKernelGenConfig cpuConfig;
  cpuConfig.simd_s = simd_s;

  cpuKernelMgr.genCPUGate(cpuConfig, getH(0), "gate_h_0");
  cpuKernelMgr.genCPUGate(cpuConfig, getH(1), "gate_h_1");
  cpuKernelMgr.genCPUGate(cpuConfig, getH(2), "gate_h_2");
  cpuKernelMgr.genCPUGate(cpuConfig, getH(3), "gate_h_3");

  cpuKernelMgr.initJIT();

  CPUStatevector<double> sv(6, simd_s);
  sv.initialize();
  suite.assertClose(sv.norm(), 1.0, "SV Initialization: Norm", GET_INFO());
  suite.assertClose(sv.prob(0), 0.0, "SV Initialization: Prob", GET_INFO());

  cpuKernelMgr.applyCPUKernel(sv.data(), sv.nQubits(), "gate_h_0");
  suite.assertClose(sv.norm(), 1.0, "Apply H at 0: Norm", GET_INFO());
  suite.assertClose(sv.prob(0), 0.5, "Apply H at 0: Prob", GET_INFO());

  sv.initialize();
  cpuKernelMgr.applyCPUKernel(sv.data(), sv.nQubits(), "gate_h_1");
  suite.assertClose(sv.norm(), 1.0, "Apply H at 1: Norm", GET_INFO());
  suite.assertClose(sv.prob(1), 0.5, "Apply H at 1: Prob", GET_INFO());

  sv.initialize();
  cpuKernelMgr.applyCPUKernel(sv.data(), sv.nQubits(), "gate_h_2");
  suite.assertClose(sv.norm(), 1.0, "Apply H at 2: Norm", GET_INFO());
  suite.assertClose(sv.prob(2), 0.5, "Apply H at 2: Prob", GET_INFO());

  sv.initialize();
  cpuKernelMgr.applyCPUKernel(sv.data(), sv.nQubits(), "gate_h_3");
  suite.assertClose(sv.norm(), 1.0, "Apply H at 3: Norm", GET_INFO());
  suite.assertClose(sv.prob(3), 0.5, "Apply H at 3: Prob", GET_INFO());

  // randomized tests
  std::vector<double> pBefore(sv.nQubits()), pAfter(sv.nQubits());
  sv.randomize();
  suite.assertClose(sv.norm(), 1.0, "SV Rand Init: Norm", GET_INFO());

  for (int q = 0; q < sv.nQubits(); q++)
    pBefore[q] = sv.prob(q);
  cpuKernelMgr.applyCPUKernel(sv.data(), sv.nQubits(), "gate_h_0");
  for (int q = 0; q < sv.nQubits(); q++)
    pAfter[q] = sv.prob(q);
  pAfter[0] = pBefore[0]; // probability could only change at the applied qubit
  suite.assertClose(sv.norm(), 1.0, "Apply H to Rand SV at 0: Norm", GET_INFO());
  suite.assertAllClose(
    pBefore, pAfter, "Apply H to Rand SV at 0: Prob", GET_INFO());

  for (int q = 0; q < sv.nQubits(); q++)
    pBefore[q] = sv.prob(q);
  cpuKernelMgr.applyCPUKernel(sv.data(), sv.nQubits(), "gate_h_1");
  for (int q = 0; q < sv.nQubits(); q++)
    pAfter[q] = sv.prob(q);
  pAfter[1] = pBefore[1]; // probability could only change at the applied qubit
  suite.assertClose(sv.norm(), 1.0, "Apply H to Rand SV at 1: Norm", GET_INFO());
  suite.assertAllClose(
    pBefore, pAfter, "Apply H to Rand SV at 1: Prob", GET_INFO());

  for (int q = 0; q < sv.nQubits(); q++)
    pBefore[q] = sv.prob(q);
  cpuKernelMgr.applyCPUKernel(sv.data(), sv.nQubits(), "gate_h_2");
  for (int q = 0; q < sv.nQubits(); q++)
    pAfter[q] = sv.prob(q);
  pAfter[2] = pBefore[2]; // probability could only change at the applied qubit
  suite.assertClose(sv.norm(), 1.0, "Apply H to Rand SV at 2: Norm", GET_INFO());
  suite.assertAllClose(
    pBefore, pAfter, "Apply H to Rand SV at 2: Prob", GET_INFO());

  for (int q = 0; q < sv.nQubits(); q++)
    pBefore[q] = sv.prob(q);
  cpuKernelMgr.applyCPUKernel(sv.data(), sv.nQubits(), "gate_h_3");
  for (int q = 0; q < sv.nQubits(); q++)
    pAfter[q] = sv.prob(q);
  pAfter[3] = pBefore[3]; // probability could only change at the applied qubit
  suite.assertClose(sv.norm(), 1.0, "Apply H to Rand SV at 3: Norm", GET_INFO());
  suite.assertAllClose(
    pBefore, pAfter, "Apply H to Rand SV at 3: Prob", GET_INFO());

  suite.displayResult();
}

void test::test_cpuH() {
  f<1>();
  f<2>();
}