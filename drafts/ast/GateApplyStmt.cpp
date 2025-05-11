#include "new_parser/Parser.h"

using namespace cast::draft;

ast::GateApplyStmt* Parser::parseGateApplyStmt() {
  // name
  assert(curToken.is(tk_Identifier) &&
         "parseGateApplyStmt expects to be called with an identifier");
  auto name = ctx.createIdentifier(curToken.toStringView());
  advance(tk_Identifier);
  
  llvm::SmallVector<ast::Expr*> params;
  llvm::SmallVector<ast::Expr*> qubits;
  // gate parameters
  if (curToken.is(tk_L_RoundBracket)) {
    advance(tk_L_RoundBracket);
    while (true) {
      auto* expr = parseExpr();
      if (expr == nullptr)
        break;
      // try to simplify the gate parameter to simple numerics
      if (auto* e = convertExprToSimpleNumeric(expr))
        expr = e;
      params.push_back(expr);
      if (curToken.is(tk_R_RoundBracket))
        break;
      requireCurTokenIs(tk_Comma, "Expect ',' to separate parameters");
      advance(tk_Comma);
      continue;
    }
    advance(tk_R_RoundBracket);
  }
  
  // target qubits
  while (true) {
    auto* expr = parseExpr();
    if (expr == nullptr)
      break;
    qubits.push_back(expr);
    // skip optional comma
    if (curToken.is(tk_Comma)) {
      advance(tk_Comma);
      continue;
    }
  }

  return new (ctx) ast::GateApplyStmt(
    name,
    ctx.createSpan(params.data(), params.size()),
    ctx.createSpan(qubits.data(), qubits.size())
  );
}

std::ostream& ast::GateApplyStmt::print(std::ostream& os) const {
  os << name;
  auto pSize = params.size();
  if (pSize > 0) {
    os << "(";
    for (size_t i = 0; i < pSize; ++i) {
      params[i]->print(os);
      if (i != pSize - 1)
        os << ", ";
    }
    os << ")";
  }
  for (const auto& qubit : qubits)
    qubit->print(os << " ");
  return os;
}

void ast::GateApplyStmt::prettyPrint(PrettyPrinter& p, int indent) const {
  p.write(indent) << getKindName() << "(" << name << "): "
                  << qubits.size() << " qubits\n";
  p.setState(indent, qubits.size());
  for (size_t i = 0; i < qubits.size(); ++i)
    qubits[i]->prettyPrint(p, indent + 1);
}