#include "cast/Legacy/QuantumGate.h"
#include "tests/TestKit.h"
#include "cast/CPU/CPUStatevector.h"

using namespace cast;
using namespace utils;

template<unsigned simd_s, unsigned nQubits>
static void internal_U1q() {
  std::stringstream ss;
  ss << "applyGate U1q (s=" << simd_s << ", nQubits=" << nQubits << ")";
  test::TestSuite suite(ss.str());

  CPUStatevector<double> sv(nQubits, simd_s);
  sv.initialize();
  for (int q = 0; q < nQubits; q++)
    sv.applyGate(legacy::QuantumGate::H(q));
  for (int q = 0; q < nQubits; q++) {
    suite.assertClose(sv.prob(q), 0.5,
      "Apply round H: Prob at qubit " + std::to_string(q), GET_INFO());
  }

  sv.randomize();
  suite.assertClose(sv.norm(), 1.0, "Rand SV: Norm", GET_INFO());

  // phase gates do not change probabilities
  for (int q = 0; q < nQubits; q++) {
    double pBefore = sv.prob(q);
    auto gate0 = legacy::QuantumGate(legacy::GateMatrix::FromName("p", {0.14}), q);
    auto gate1 = legacy::QuantumGate(legacy::GateMatrix::FromName("p", {0.41}), (q+1) % nQubits);
    auto gate = gate0.lmatmul(gate1);

    sv.applyGate(gate);
    std::stringstream ss;
    ss << "Phase gate at qubits "
       << q << " " << ((q+1) % nQubits);
    suite.assertClose(sv.norm(), 1.0, ss.str() + ": Norm", GET_INFO());
    suite.assertClose(pBefore, sv.prob(q), ss.str() + ": Prob", GET_INFO());
  }

  suite.displayResult();
}

void test::test_applyGate() {
  internal_U1q<1, 8>();
  internal_U1q<2, 8>();
}