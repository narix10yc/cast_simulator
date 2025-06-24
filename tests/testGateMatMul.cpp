#include "cast/Legacy/QuantumGate.h"
#include "tests/TestKit.h"
#include "cast/CPU/CPUStatevector.h"

using namespace cast;
using namespace cast::test;
using namespace utils;

static void basics() {
  TestSuite suite("MatMul between Gates Basics");
  legacy::QuantumGate gate0, gate1;
  double norm;
  gate0 = legacy::QuantumGate::I1(1);
  gate1 = legacy::QuantumGate::I1(1);
  norm = utils::maximum_norm(
    *gate0.lmatmul(gate1).gateMatrix.getConstantMatrix(),
    legacy::GateMatrix::MatrixI1_c);
  suite.assertClose(norm, 0.0, "I multiply by I Same Qubit", GET_INFO());

  gate0 = legacy::QuantumGate::I1(2);
  norm = utils::maximum_norm(
    *gate0.lmatmul(gate1).gateMatrix.getConstantMatrix(),
    legacy::GateMatrix::MatrixI2_c);
  suite.assertClose(norm, 0.0, "I multiply by I Different Qubit", GET_INFO());

  gate1 = legacy::QuantumGate(legacy::GateMatrix(utils::randomUnitaryMatrix(2)), 2);
  norm = utils::maximum_norm(
    *gate0.lmatmul(gate1).gateMatrix.getConstantMatrix(),
    *gate1.gateMatrix.getConstantMatrix());
  suite.assertClose(norm, 0.0, "I multiply by U Same Qubit", GET_INFO());

  suite.displayResult();
}

template<CPUSimdWidth SimdWidth, int nQubits>
static void internal() {
  std::stringstream titleSS;
  titleSS << "MatMul between Gates (s=" << SimdWidth
          << ", nQubits=" << nQubits << ")";
  TestSuite suite(titleSS.str());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> d(0, nQubits - 1);
  for (int i = 0; i < 3; ++i) {
    int a = d(gen);
    int b = d(gen);
    auto gate0 = legacy::QuantumGate::RandomUnitary(a);
    auto gate1 = legacy::QuantumGate::RandomUnitary(b);
    auto gate = gate0.lmatmul(gate1);

    cast::CPUStatevector<double> sv0(nQubits, SimdWidth),
                                 sv1(nQubits, SimdWidth);
    sv0.randomize();
    sv1 = sv0;

    sv0.applyGate(gate0).applyGate(gate1);
    sv1.applyGate(gate);

    std::stringstream ss;
    ss << "Apply U gate on qubits " << a << " and " << b;
    suite.assertClose(sv0.norm(), 1.0, ss.str() + ": Separate Norm", GET_INFO());
    suite.assertClose(sv1.norm(), 1.0, ss.str() + ": Joint Norm", GET_INFO());
    suite.assertClose(fidelity(sv0, sv1), 1.0,
      ss.str() + ": Fidelity", GET_INFO());
  }
  suite.displayResult();
}

void cast::test::test_gateMatMul() {
  basics();
  internal<W128, 4>();
  internal<W256, 8>();
}