#include "cast/Core/QuantumGate.h"
#include "tests/TestKit.h"
#include <llvm/Support/Casting.h>

using namespace cast;

static SuperopQuantumGatePtr getSOGateForSPC(double p, int q) {
  constexpr double a = 2.0 / 3.0;
  constexpr double b = 4.0 / 3.0;
  // clang-format off
  ComplexSquareMatrix mat{
    // real part
    {1 - a * p,       0.0,       0.0,     a * p,
           0.0, 1 - b * p,       0.0,       0.0,
           0.0,       0.0, 1 - b * p,       0.0,
         a * p,       0.0,       0.0, 1 - a * p},
    // imag part
    {0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0}
  };
  // clang-format on

  auto scalarGM = std::make_shared<ScalarGateMatrix>(std::move(mat));
  return SuperopQuantumGate::Create(scalarGM, {q});
}

bool cast::test::test_quantumChannel() {
  test::TestSuite suite("QuantumChannel SPC superoperator mapping");

  auto gate = StandardQuantumGate::I1(0);
  gate->setNoiseSPC(0.1);
  auto soGate = gate->getSuperopGate();

  auto targetSOGate = getSOGateForSPC(0.1, 0);
  suite.assertCloseFP64(cast::maximum_norm(soGate->getMatrix()->matrix(),
                                           targetSOGate->getMatrix()->matrix()),
                        0.0,
                        "SPC superoperator matches analytic reference",
                        GET_INFO());

  return suite.displayResult();
}
