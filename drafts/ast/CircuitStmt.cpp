#include "new_parser/Parser.h"

using namespace cast::draft;

std::ostream& ast::CircuitStmt::print(std::ostream& os) const {
  os << "Circuit";
  if (attr != nullptr)
    attr->print(os);
  os << " " << name << " {\n";
  for (const auto& stmt : body)
    stmt->print(os << "  ") << "\n";
  os << "}\n";
  return os;
}

void ast::CircuitStmt::prettyPrint(PrettyPrinter& p, int indent) const {
  unsigned size = body.size();
  p.write(indent) << getKindName() << "(" << name << "): "
                  << size << " stmts\n";
  if (attr != nullptr)
    p.os << "Attr\n";
  p.setState(indent, size);
  for (unsigned i = 0; i < size; ++i)
    body[i]->prettyPrint(p, indent + 1);
}
