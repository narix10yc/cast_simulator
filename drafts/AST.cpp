#include "new_parser/Parser.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

using namespace cast::draft;

std::ostream& ast::Attribute::print(std::ostream& os) const {
  if (nQubits == 0 && nParams == 0 && phase.getValue() == 0.0)
    return os;
  os << "[";
  bool needComma = false;
  if (nQubits != 0) {
    os << "nqubits=" << nQubits;
    needComma = true;
  }
  if (nParams != 0) {
    if (needComma)
      os << ", ";
    os << "nparams=" << nParams;
    needComma = true;
  }
  if (phase.getValue() != 0.0) {
    if (needComma)
      os << ", ";
    os << "phase=";
    phase.print(os);
  }
  return os << "]";
}

std::ostream& ast::SimpleNumericExpr::print(std::ostream& os) const {
  if (!_value.holdingValue())
    return os << "0";
  if (_value.is<int>())
    return os << _value.get<int>();
  if (_value.is<double>())
    return os << _value.get<double>();
    
  assert(_value.is<FractionPi>() && "Invalid state");
  const auto& fraction = _value.get<FractionPi>();

  assert(fraction.numerator != 0 && "Numerator is zero?");
  assert(fraction.denominator != 0 && "Invalid state");

  if (fraction.numerator == 1)
    os << "Pi";
  else if (fraction.numerator == -1)
    os << "-Pi";
  else
    os << fraction.numerator << "*Pi";
  if (fraction.denominator != 1)
    os << '/' << fraction.denominator;
  return os;
}

std::ostream& ast::GateApplyStmt::print(std::ostream& os) const {
  os << name;
  if (attribute)
    attribute->print(os);
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
  assert(attribute == nullptr && "GateChainStmt should not have attribute");
  for (size_t i = 0, size = gates.size(); i < size; ++i) {
    gates[i].print(os);
    os << ((i == size - 1) ? ";" : "\n@ ");
  }
  return os;
}

std::ostream& ast::CircuitStmt::print(std::ostream& os) const {
  os << "Circuit";
  if (attribute)
    attribute->print(os);
  os << " " << name << " {\n";
  for (const auto& stmt : body)
    stmt->print(os << "  ") << "\n";
  os << "}\n";
  return os;
}

std::ostream& ast::RootNode::print(std::ostream& os) const {
  for (const auto& stmt : stmts) {
    stmt->print(os);
    os << '\n';
  }
  return os;
}