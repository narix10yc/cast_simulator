#include "saot/QuantumGate.h"
#include "tests/TestKit.h"
#include "utils/statevector.h"

using namespace saot;
using namespace saot::test;
using namespace utils;

static void basics() {
  TestSuite suite("MatMul between Gates Basics");
  QuantumGate gate0, gate1;
  double norm;
  gate0 = QuantumGate::I1(1);
  gate1 = QuantumGate::I1(1);
  norm = utils::maximum_norm(
    *gate0.lmatmul(gate1).gateMatrix.getConstantMatrix(),
    GateMatrix::MatrixI1_c);
  suite.assertClose(norm, 0.0, "I multiply by I Same Qubit", GET_INFO());

  gate0 = QuantumGate::I1(2);
  norm = utils::maximum_norm(
    *gate0.lmatmul(gate1).gateMatrix.getConstantMatrix(),
    GateMatrix::MatrixI2_c);
  suite.assertClose(norm, 0.0, "I multiply by I Different Qubit", GET_INFO());

  gate1 = QuantumGate(GateMatrix(utils::randomUnitaryMatrix(2)), 2);
  norm = utils::maximum_norm(
    *gate0.lmatmul(gate1).gateMatrix.getConstantMatrix(),
    *gate1.gateMatrix.getConstantMatrix());
  suite.assertClose(norm, 0.0, "I multiply by U Same Qubit", GET_INFO());

  suite.displayResult();
}

template<unsigned nqubits, unsigned simd_s>
static void internal() {
  std::stringstream titleSS;
  titleSS << "MatMul between Gates (s=" << simd_s
          << ", nqubits=" << nqubits << ")";
  TestSuite suite(titleSS.str());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> d(0, nqubits - 1);
  for (int i = 0; i < 3; ++i) {
    int a = d(gen);
    int b = d(gen);
    auto gate0 = QuantumGate::RandomUnitary(a);
    auto gate1 = QuantumGate::RandomUnitary(b);
    auto gate = gate0.lmatmul(gate1);

    utils::StatevectorAlt<double> sv0(nqubits, simd_s), sv1(nqubits, simd_s);
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

void saot::test::test_gateMatMul() {
  basics();
  internal<4, 1>();
  internal<8, 2>();
}