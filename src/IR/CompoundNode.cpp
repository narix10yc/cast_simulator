#include "cast/IR/IRNode.h"

using namespace cast::ir;

std::ostream& CompoundNode::print(std::ostream& os, int indent) const {
  for (const auto& node : nodes)
    node->print(os, indent);
  return os;
} // CompoundNode::print