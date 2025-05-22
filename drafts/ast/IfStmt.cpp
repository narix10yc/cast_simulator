#include "new_parser/Parser.h"

using namespace cast::draft::ast;

IfStmt* Parser::parseIfStmt() {
  assert(curToken.is(tk_If) &&
         "parseIfStmt expects to be called with an 'If' token");
  advance(tk_If);

  requireCurTokenIs(tk_L_RoundBracket, "Expect '(' after 'If'");
  advance(tk_L_RoundBracket);
  auto* condition = parseExpr();
  if (condition == nullptr) {
    logErrHere("Expect a condition expression");
    failAndExit();
  }
  requireCurTokenIs(tk_R_RoundBracket, "Expect ')' after condition");
  advance(tk_R_RoundBracket);

  auto body = parseCircuitLevelStmtList();
  std::span<Stmt*> elseBody;

  if (curToken.is(tk_Else)) {
    advance(tk_Else);
    elseBody = parseCircuitLevelStmtList();
  }
  return new (ctx) IfStmt(condition, body, elseBody);
}

std::ostream& IfStmt::print(std::ostream& os) const {
  os << "If (";
  condition->print(os) << ") {\n";
  for (auto* s : body)
    s->print(os << "  ");
  os << "\n} Else {\n";
  for (auto* s : elseBody)
    s->print(os << "  ");
  return os << "\n}\n";
}

void IfStmt::prettyPrint(PrettyPrinter& p, int indent) const {
  p.write(indent) << getKindName() << "\n";
  p.setState(indent, elseBody.empty() ? 2 : 3);

  p.setPrefix("condition").write(indent + 1);
  condition->print(p.os) << "\n";

  p.setPrefix("body").write(indent + 1) << body.size() << " stmts\n";
  p.setState(indent + 1, body.size());
  for (auto* s : body)
    s->prettyPrint(p, indent + 2);
  if (!elseBody.empty()) {
    p.setPrefix("elsebody").write(indent + 1) << elseBody.size() << " stmts\n";
    p.setState(indent + 1, elseBody.size());
    for (auto* s : elseBody)
      s->prettyPrint(p, indent + 2);
  }
}