#include "new_parser/Parser.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

#include "cast/CircuitGraph.h"

using namespace cast::draft;

void ast::CircuitStmt::toCircuitGraph(cast::CircuitGraph& graph) const {
  assert(false && "Not implemented yet");
}

std::ostream& ast::Attribute::print(std::ostream& os) const {
  os << "<";
  bool needComma = false;
  if (nQubits != nullptr) {
    nQubits->print(os << "nqubits=");
    needComma = true;
  }
  if (nParams != nullptr) {
    if (needComma)
      os << ", ";
    nParams->print(os << "nparams=");
    needComma = true;
  }
  if (phase != nullptr) {
    if (needComma)
      os << ", ";
    phase->print(os << "phase=");
  }
  return os << ">";
}

std::ostream& ast::ParameterDeclExpr::print(std::ostream& os) const {
  os << "(";
  utils::printSpanWithPrinterNoBracket(
    os, parameters, 
    [](std::ostream& os, ast::IdentifierExpr* param) {
      param->print(os);
    }
  );
  return os << ") ";
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

std::ostream& ast::GateChainStmt::print(std::ostream& os) const {
  for (size_t i = 0, size = gates.size(); i < size; ++i) {
    gates[i]->print(os);
    os << ((i == size - 1) ? ";" : "\n@ ");
  }
  return os;
}

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

std::ostream& ast::RootNode::print(std::ostream& os) const {
  for (const auto* stmt : stmts) {
    stmt->print(os);
    os << '\n';
  }
  return os;
}

ast::CircuitStmt* ast::RootNode::lookupCircuit(const std::string& name) {
  for (auto& stmt : stmts) {
    if (auto* circuit = llvm::dyn_cast<CircuitStmt>(stmt)) {
      if (name.empty() || circuit->name.str == name)
        return circuit;
    }
  }
  return nullptr;
}