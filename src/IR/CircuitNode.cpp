#include "cast/IR/IRNode.h"

using namespace cast::ir;

std::ostream& CircuitNode::print(std::ostream& os) const {
  os << "cast.circuit " << name << ") {\n";
  body.print(os);
  os << "}\n";
  return os;
} // CircuitNode::print