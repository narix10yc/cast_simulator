#include "cast/Core/IRNode.h"

using namespace cast::ir;

std::ostream& CompoundNode::print(std::ostream& os, int indent) const {
  for (const auto& node : nodes)
    node->print(os, indent);
  return os;
} // CompoundNode::print

std::ostream& 
CompoundNode::impl_visualize(std::ostream& os, int width, int n_qubits) const {
  for (const auto& node : nodes)
    node->impl_visualize(os, width, n_qubits);
  return os;
}