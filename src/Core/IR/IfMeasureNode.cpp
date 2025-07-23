#include "cast/Core/IRNode.h"

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

std::ostream&
IfMeasureNode::impl_visualize(std::ostream& os, int width, int n_qubits) const {
  for (unsigned i = 0; i < qubit * width; ++i)
    os.put('=');
  for (unsigned i = 0; i < width / 2; ++i)
    os.put(' ');
  os << "M";
  for (unsigned i = 0; i < width / 2; ++i)
    os.put(' ');
  for (unsigned i = 0; i < (n_qubits - qubit - 1) * (width + 1); ++i)
    os.put('=');

  // then block
  os.write(" Then\n", 6);
  thenBody.impl_visualize(os, width, n_qubits);

  // separation
  for (unsigned i = 0; i < n_qubits * width + (n_qubits - 1); ++i)
    os.put('=');

  // else block
  os.write(" Else\n", 6);
  elseBody.impl_visualize(os, width, n_qubits);

  // separation
  for (unsigned i = 0; i < n_qubits * width + (n_qubits - 1); ++i)
    os.put('=');

  os.write(" Join\n", 6);
  return os;
}