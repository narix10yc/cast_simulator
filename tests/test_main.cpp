#include "tests/TestKit.h"
#include "utils/statevector.h"
#include "saot/QuantumGate.h"
#include "simulation/KernelGen.h"
#include "simulation/JIT.h"
#include "utils/iocolor.h"

using namespace saot;
using namespace saot::test;
using namespace utils::statevector;

using namespace llvm;

#define FUNC_TYPE void(void*, uint64_t, uint64_t, void*)

// testH
#include "test_h.inc"

template<unsigned simdS>
void testU() {
  TestSuite suite("Gate U with simdS = " + std::to_string(simdS));
  suite.assertClose(1e-10, 0.0, GET_INFO("1e-10 is close to 0.0"));

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = std::make_unique<llvm::Module>("myModule", *llvmContext);

  CPUKernelGenConfig cpuConfig;
  cpuConfig.simdS = simdS;
  cpuConfig.forceDenseKernel = true;
  cpuConfig.matrixLoadMode = CPUKernelGenConfig::StackLoadMatElems;

  genCPUCode(*llvmModule, cpuConfig, {GateMatrix::MatrixI1_c, 0}, "gate_u_0");
  genCPUCode(*llvmModule, cpuConfig, {GateMatrix::MatrixI1_c, 1}, "gate_u_1");
  genCPUCode(*llvmModule, cpuConfig, {GateMatrix::MatrixI1_c, 2}, "gate_u_2");
  genCPUCode(*llvmModule, cpuConfig, {GateMatrix::MatrixI1_c, 3}, "gate_u_3");

  auto jit = saot::createJITSession();
  cantFail(jit->addIRModule(llvm::orc::ThreadSafeModule(
    std::move(llvmModule), std::move(llvmContext))));

  auto f_h0 = jit->lookup("gate_u_0")->toPtr<FUNC_TYPE>();
  auto f_h1 = jit->lookup("gate_u_1")->toPtr<FUNC_TYPE>();
  auto f_h2 = jit->lookup("gate_u_2")->toPtr<FUNC_TYPE>();
  auto f_h3 = jit->lookup("gate_u_3")->toPtr<FUNC_TYPE>();

  StatevectorAlt<double, simdS> sv(/* nqubits */ 6, /* initialize */ true);
  suite.assertClose(sv.norm(), 1.0, GET_INFO("SV Initialization: Norm"));
  suite.assertClose(sv.prob(0), 0.0, GET_INFO("SV Initialization: Prob"));

  f_h0(sv.data, 0ULL, 1ULL << (sv.nqubits - 1 - cpuConfig.simdS), nullptr);
  suite.assertClose(sv.norm(), 1.0, GET_INFO("Apply H at 0: Norm"));
  suite.assertClose(sv.prob(0), 0.5, GET_INFO("Apply H at 0: Prob"));
  
  sv.initialize();
  f_h1(sv.data, 0ULL, 1ULL << (sv.nqubits - 1 - cpuConfig.simdS), nullptr);
  suite.assertClose(sv.norm(), 1.0, GET_INFO("Apply H at 1: Norm"));
  suite.assertClose(sv.prob(1), 0.5, GET_INFO("Apply H at 1: Prob"));

  sv.initialize();
  f_h2(sv.data, 0ULL, 1ULL << (sv.nqubits - 1 - cpuConfig.simdS), nullptr);
  suite.assertClose(sv.norm(), 1.0, GET_INFO("Apply H at 2: Norm"));
  suite.assertClose(sv.prob(2), 0.5, GET_INFO("Apply H at 2: Prob"));

  sv.initialize();
  f_h3(sv.data, 0ULL, 1ULL << (sv.nqubits - 1 - cpuConfig.simdS), nullptr);
  suite.assertClose(sv.norm(), 1.0, GET_INFO("Apply H at 3: Norm"));
  suite.assertClose(sv.prob(3), 0.5, GET_INFO("Apply H at 3: Prob"));

  // randomized tests
  std::vector<double> pBefore(sv.nqubits), pAfter(sv.nqubits);
  sv.randomize();
  suite.assertClose(sv.norm(), 1.0, GET_INFO("SV Rand Init: Norm"));

  for (int q = 0; q < sv.nqubits; q++)
    pBefore[q] = sv.prob(q);
  f_h0(sv.data, 0ULL, 1ULL << (sv.nqubits - 1 - cpuConfig.simdS), nullptr);
  for (int q = 0; q < sv.nqubits; q++)
    pAfter[q] = sv.prob(q);
  pAfter[0] = pBefore[0]; // probably could only change at the applied qubit
  suite.assertClose(sv.norm(), 1.0, GET_INFO("Apply H to Rand SV at 0: Norm"));
  suite.assertAllClose(
    pBefore, pAfter, GET_INFO("Apply H to Rand SV at 0: Prob"));

  for (int q = 0; q < sv.nqubits; q++)
    pBefore[q] = sv.prob(q);
  f_h1(sv.data, 0ULL, 1ULL << (sv.nqubits - 1 - cpuConfig.simdS), nullptr);
  for (int q = 0; q < sv.nqubits; q++)
    pAfter[q] = sv.prob(q);
  pAfter[1] = pBefore[1]; // probably could only change at the applied qubit
  suite.assertClose(sv.norm(), 1.0, GET_INFO("Apply H to Rand SV at 1: Norm"));
  suite.assertAllClose(
    pBefore, pAfter, GET_INFO("Apply H to Rand SV at 1: Prob"));

  for (int q = 0; q < sv.nqubits; q++)
    pBefore[q] = sv.prob(q);
  f_h2(sv.data, 0ULL, 1ULL << (sv.nqubits - 1 - cpuConfig.simdS), nullptr);
  for (int q = 0; q < sv.nqubits; q++)
    pAfter[q] = sv.prob(q);
  pAfter[2] = pBefore[2]; // probably could only change at the applied qubit
  suite.assertClose(sv.norm(), 1.0, GET_INFO("Apply H to Rand SV at 2: Norm"));
  suite.assertAllClose(
    pBefore, pAfter, GET_INFO("Apply H to Rand SV at 2: Prob"));

  for (int q = 0; q < sv.nqubits; q++)
    pBefore[q] = sv.prob(q);
  f_h3(sv.data, 0ULL, 1ULL << (sv.nqubits - 1 - cpuConfig.simdS), nullptr);
  for (int q = 0; q < sv.nqubits; q++)
    pAfter[q] = sv.prob(q);
  pAfter[3] = pBefore[3]; // probably could only change at the applied qubit
  suite.assertClose(sv.norm(), 1.0, GET_INFO("Apply H to Rand SV at 3: Norm"));
  suite.assertAllClose(
    pBefore, pAfter, GET_INFO("Apply H to Rand SV at 3: Prob"));

  suite.displayResult();
}

int main() {
  testH</* simdS */ 1>();
  testH</* simdS */ 2>();

  testU</* simdS */ 1>();
  testU</* simdS */ 2>();

  return 0;
}