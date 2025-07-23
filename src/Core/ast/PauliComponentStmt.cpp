#include "cast/Core/AST/Parser.h"

using namespace cast::ast;

PauliComponentStmt* Parser::parsePauliComponentStmt() {
  assert(curToken.is(tk_Identifier) &&
         "parsePauliComponentStmt expects to be called with an identifier");
  auto str = ctx.createIdentifier(curToken.toStringView());
  advance(tk_Identifier);
  auto* weight = parseExpr();
  if (weight == nullptr) {
    logErrHere("Expect a weight after the Pauli component");
    failAndExit();
  }
  if (curToken.isNot(tk_Semicolon)) {
    logErrHere("Expect ';' to end a Pauli component");
    failAndExit();
  }
  advance(tk_Semicolon);
  return new (ctx) PauliComponentStmt(str, weight);
}

std::ostream& PauliComponentStmt::print(std::ostream& os) const {
  os << str << " ";
  assert(weight != nullptr && "weight should not be nullptr");
  weight->print(os);
  return os << ";";
}

void PauliComponentStmt::prettyPrint(PrettyPrinter& p, int indent) const {
  p.write(indent) << getKindName() << ": " << str << "\n";
  assert(weight != nullptr && "weight should not be nullptr");
  p.setState(indent, 1);
  p.setPrefix("weight: ");
  weight->prettyPrint(p, indent + 1);
}