#include "cast/IR/IRNode.h"

using namespace cast::ir;

std::ostream& CircuitNode::print(std::ostream& os, int indent) const {
  writeIndent(os, indent) << "cast.circuit(" << name << ") {\n";
  body.print(os, indent + 1);
  writeIndent(os, indent) << "}\n";
  return os;
} // CircuitNode::print