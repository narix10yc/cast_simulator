#include "cast/Core/AST/Parser.h"

using namespace cast::ast;

int MinusOpExpr::getPrecedence() { return 30; }

std::ostream& MinusOpExpr::print(std::ostream& os) const {
  os << "-";
  operand->print(os);
  return os;
}

void MinusOpExpr::prettyPrint(PrettyPrinter& p, int indent) const {
  p.write(indent) << getKindName() << "\n";
  p.setState(indent, 1);
  p.setPrefix("operand: ");
  operand->prettyPrint(p, indent + 1);
}