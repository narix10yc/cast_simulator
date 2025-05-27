#include "cast/IR/IRNode.h"

using namespace cast::ir;

std::ostream& IfMeasureNode::print(std::ostream& os, int indent) const {
  writeIndent(os, indent) << "cast.if_measure(" << qubit << ") {\n";
  for (const auto& node : thenBody.nodes)
    node->print(os, indent + 1);
  writeIndent(os, indent) << "}\n";
  writeIndent(os, indent) << "else {\n";
  for (const auto& node : elseBody.nodes)
    node->print(os, indent + 1);
  writeIndent(os, indent) << "}\n";
  return os;
} // IfMeasureNode::print