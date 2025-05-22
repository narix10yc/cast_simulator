#include "cast/IR/IRNode.h"

using namespace cast::ir;

std::ostream& CompoundNode::print(std::ostream& os) const {
  for (const auto& node : nodes)
    node->print(os) << "\n";
  return os;
} // CompoundNode::print