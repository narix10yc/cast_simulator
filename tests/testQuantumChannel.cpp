#include "cast/Core/QuantumGate.h"
#include "tests/TestKit.h"
#include "llvm/Support/Casting.h"

using namespace cast;

static SuperopQuantumGatePtr getSOGateForSPC(double p, int q) {
  constexpr double a = 2.0 / 3.0;
  constexpr double b = 4.0 / 3.0;
  ComplexSquareMatrix mat{// real part
                          {1 - a * p,
                           0,
                           0,
                           a * p,
                           0,
                           1 - b * p,
                           0,
                           0,
                           0,
                           0,
                           1 - b * p,
                           0,
                           a * p,
                           0,
                           0,
                           1 - a * p},
                          // imag part
                          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  auto scalarGM = std::make_shared<ScalarGateMatrix>(std::move(mat));
  return SuperopQuantumGate::Create(scalarGM, {q});
}

void cast::test::test_quantumChannel() {
  test::TestSuite suite("QuantumChannel Test Suite");

  auto gate = StandardQuantumGate::I1(0);
  gate->setNoiseSPC(0.1);

  auto soGate = gate->getSuperopGate();

  auto targetSOGate = getSOGateForSPC(0.1, 0);
  suite.assertCloseF64(cast::maximum_norm(soGate->getMatrix()->matrix(),
                                       targetSOGate->getMatrix()->matrix()),
                    0.0,
                    "Superop gate SPC test",
                    GET_INFO());

  suite.displayResult();
}