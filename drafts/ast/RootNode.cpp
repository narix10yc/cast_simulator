#include "new_parser/Parser.h"

using namespace cast::draft;

std::ostream& ast::RootNode::print(std::ostream& os) const {
  for (const auto* stmt : stmts) {
    stmt->print(os);
    os << '\n';
  }
  return os;
}

void ast::RootNode::prettyPrint(PrettyPrinter& p, int indent) const {
  unsigned size = stmts.size();
  p.write(indent) << getKindName() << ": " << size << " stmts\n";
  p.setState(indent, size);
  for (unsigned i = 0; i < size; ++i)
    stmts[i]->prettyPrint(p, indent+1);
}