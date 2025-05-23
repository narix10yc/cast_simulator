#include "cast/Transform/Transform.h"

using namespace cast;
using namespace cast::draft;

GateMatrixPtr transform::convertGate(ast::GateApplyStmt* astGate,
                                     ast::ASTContext& astCtx) {
  std::string name(astGate->name.str);

  // TODO: use astCtx to print error messages
  const auto assertGate = [astGate](int nQubits, int nParams) {
    assert(astGate->qubits.size() == nQubits &&
           "Invalid number of qubits in gate");
    assert(astGate->params.size() == nParams &&
           "Invalid number of parameters in gate");
  };

  // Single qubit gates
  if (name == "X") {
    assertGate(1, 0);
    return ScalarGateMatrix::X();
  }
  if (name == "Y") {
    assertGate(1, 0);
    return ScalarGateMatrix::Y();
  }
  if (name == "Z") {
    assertGate(1, 0);
    return ScalarGateMatrix::Z();
  }
  if (name == "H") {
    assertGate(1, 0);
    return ScalarGateMatrix::H();
  }
  if (name == "RX") {
    assertGate(1, 1);
    auto* theta = ast::reduceExprToSimpleNumeric(astCtx, astGate->params[0]);
    assert(theta != nullptr && "Only supporting scalar values for now");
    return ScalarGateMatrix::RX(theta->getValue());
  }
  if (name == "RY") {
    assertGate(1, 1);
    auto* theta = ast::reduceExprToSimpleNumeric(astCtx, astGate->params[0]);
    assert(theta != nullptr && "Only supporting scalar values for now");
    return ScalarGateMatrix::RY(theta->getValue());
  }
  if (name == "RZ") {
    assertGate(1, 1);
    auto* phi = ast::reduceExprToSimpleNumeric(astCtx, astGate->params[0]);
    assert(phi != nullptr && "Only supporting scalar values for now");
    return ScalarGateMatrix::RZ(phi->getValue());
  }

  // Two qubit gates
  if (name == "CX" || name == "CNOT") {
    assertGate(2, 0);
    return ScalarGateMatrix::CX();
  }

  std::cerr << "Not implemented or unsupported gate: " << name << "\n";
  assert(false && "Not implemented or unsupported gate");
  return nullptr;
}