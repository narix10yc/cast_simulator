#include "cast/IR/IRNode.h"

using namespace cast::ir;

std::ostream& IfMeasureNode::print(std::ostream& os) const {
  os << "cast.if_measure(" << qubit << ") {\n";
  for (const auto& node : thenBody.nodes)
    node->print(os);
  os << "}\n";
  os << "else {\n";
  for (const auto& node : elseBody.nodes)
    node->print(os);
  os << "}\n";
  return os;
} // IfMeasureNode::print