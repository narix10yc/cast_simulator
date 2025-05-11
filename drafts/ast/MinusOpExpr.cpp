#include "new_parser/Parser.h"

using namespace cast::draft;

int ast::MinusOpExpr::getPrecedence() {
  return 30;
}

std::ostream& ast::MinusOpExpr::print(std::ostream& os) const {
  os << "-";
  operand->print(os);
  return os;
}

void ast::MinusOpExpr::prettyPrint(PrettyPrinter& p, int indent) const {
  p.write(indent) << getKindName() << "\n";
  p.setState(indent, 1);
  p.setPrefix("operand: ");
  operand->prettyPrint(p, indent + 1);
}