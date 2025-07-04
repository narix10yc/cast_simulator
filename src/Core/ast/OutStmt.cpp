#include "cast/Core/AST/Parser.h"

using namespace cast::ast;

OutStmt* Parser::parseOutStmt() {
  assert(curToken.is(tk_Out) &&
         "parseOutStmt expects to be called with an 'Out' token");
  advance(tk_Out);
  // expr could be null;
  auto* expr = parseExpr();
  advance(tk_Semicolon);
  return new (ctx) OutStmt(expr);
}

std::ostream& OutStmt::print(std::ostream& os) const {
  os << "Out";
  if (expr != nullptr)
    expr->print(os << "(") << ")";
  return os << ";";
}

void OutStmt::prettyPrint(PrettyPrinter& p, int indent) const {
  p.write(indent) << getKindName() << "\n";
  if (expr == nullptr)
    return;
  p.setState(indent, 1);
  p.setPrefix("expr").write(indent + 1);
  expr->print(p.os) << "\n";
}
