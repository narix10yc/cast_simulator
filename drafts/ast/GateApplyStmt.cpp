#include "new_parser/Parser.h"

using namespace cast::draft;

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