#include "cast/IR/IRNode.h"
#include "llvm/Support/Casting.h"
#include "utils/iocolor.h"

using namespace cast::ir;

std::ostream& CircuitNode::print(std::ostream& os, int indent) const {
  writeIndent(os, indent) << "cast.circuit(" << name << ") {\n";
  body.print(os, indent + 1);
  writeIndent(os, indent) << "}\n";
  return os;
}

std::ostream& CircuitNode::displayInfo(std::ostream& os, int verbose) const {
  os << BOLDCYAN("=== Info of ir::CircuitNode @ " << this << " === ")
     << "(Verbose " << verbose << ")\n";

  os << CYAN("- name: ") << name << "\n";
  os << CYAN("- Num CircuitGraph: ") << countNumCircuitGraphs() << "\n";

  os << BOLDCYAN("====================================") << "\n";
  return os;
}

// implementation of countNumCircuitGraphs
namespace {
  unsigned impl_countNumCircuitGraphs(IRNode* node);

  unsigned impl_countNumCircuitGraphs(const CompoundNode& compoundNode) {
    unsigned count = 0;
    for (const auto& node : compoundNode.nodes)
      count += impl_countNumCircuitGraphs(node.get());
    return count;
  }

  unsigned impl_countNumCircuitGraphs(IRNode* node) {
    if (llvm::isa<CircuitGraphNode>(node))
      return 1;
    if (auto* ifNode = llvm::dyn_cast<IfMeasureNode>(node)) {
      return impl_countNumCircuitGraphs(ifNode->thenBody) +
             impl_countNumCircuitGraphs(ifNode->elseBody);
    }
    assert(false && "Unknown node type");
    return 0;
  }
}

unsigned CircuitNode::countNumCircuitGraphs() const {
  return impl_countNumCircuitGraphs(body);
}