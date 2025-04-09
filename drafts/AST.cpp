#include "new_parser/Parser.h"
#include "utils/iocolor.h"

using namespace cast::draft;

std::ostream& ast::Attribute::print(std::ostream& os) const {
  os << "[nqubits=" << nQubits
    << ", nparams=" << nParams
    << ", phase=";
  phase.print(os) << "]";
  return os;
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
    os << "pi";
  else if (fraction.numerator == -1)
    os << "-pi";
  else
    os << fraction.numerator << "*pi";
  if (fraction.denominator != 1)
    os << '/' << fraction.denominator;
  return os;
}

std::ostream& ast::CircuitStmt::print(std::ostream& os) const {
  os << "circuit";
  if (attribute)
    attribute->print(os);
  os << " " << name << " {}";
  return os;
}

std::ostream& ast::RootNode::print(std::ostream& os) const {
  for (const auto& stmt : stmts) {
    stmt->print(os);
    os << '\n';
  }
  return os;
}