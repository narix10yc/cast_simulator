#include "cast/Transform/Transform.h"

using namespace cast;
using namespace cast::draft;

GateMatrixPtr transform::convertGate(ast::GateApplyStmt* astGate,
                                     ast::ASTContext& astCtx) {
  std::string name(astGate->name.str);
  if (name == "X") {
    assert(astGate->params.size() == 0);
    return ScalarGateMatrix::X();
  }
  if (name == "Y") {
    assert(astGate->params.size() == 0);
    return ScalarGateMatrix::Y();
  }
  if (name == "Z") {
    assert(astGate->params.size() == 0);
    return ScalarGateMatrix::Z();
  }
  if (name == "H") {
    assert(astGate->params.size() == 0);
    return ScalarGateMatrix::H();
  }
  if (name == "RX") {
    assert(astGate->params.size() == 1);
    auto* theta = ast::reduceExprToSimpleNumeric(astCtx, astGate->params[0]);
    assert(theta != nullptr && "Only supporting scalar values for now");
    return ScalarGateMatrix::RX(theta->getValue());
  }
  if (name == "RY") {
    assert(astGate->params.size() == 1);
    auto* theta = ast::reduceExprToSimpleNumeric(astCtx, astGate->params[0]);
    assert(theta != nullptr && "Only supporting scalar values for now");
    return ScalarGateMatrix::RY(theta->getValue());
  }
  if (name == "RZ") {
    assert(astGate->params.size() == 1);
    auto* phi = ast::reduceExprToSimpleNumeric(astCtx, astGate->params[0]);
    assert(phi != nullptr && "Only supporting scalar values for now");
    return ScalarGateMatrix::RZ(phi->getValue());
  }

  std::cerr << "Not implemented or unsupported gate: " << name << "\n";
  assert(false && "Not implemented or unsupported gate");
  return nullptr;
}