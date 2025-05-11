#include "new_parser/Parser.h"

using namespace cast::draft;

std::ostream& ast::GateChainStmt::print(std::ostream& os) const {
  for (size_t i = 0, size = gates.size(); i < size; ++i) {
    gates[i]->print(os);
    os << ((i == size - 1) ? ";" : "\n@ ");
  }
  return os;
}

void ast::GateChainStmt::prettyPrint(PrettyPrinter& p, int indent) const {
  p.write(indent) << getKindName() << ": " << gates.size() << " gates\n";
  p.setState(indent, gates.size());
  for (size_t i = 0; i < gates.size(); ++i)
    gates[i]->prettyPrint(p, indent + 1);
}