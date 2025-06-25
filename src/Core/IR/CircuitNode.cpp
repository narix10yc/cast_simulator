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

std::ostream& CircuitNode::impl_visualize(std::ostream& os,
                                      int width, 
                                      int n_qubits) const {
  return body.impl_visualize(os, width, n_qubits);
}

std::ostream& CircuitNode::visualize(std::ostream& os) const {
  // Some circuit graphs may have fewer qubits
  int nQubits = 0;
  for (const auto& graph : getAllCircuitGraphs())
    nQubits = std::max(nQubits, graph->nQubits());
  return impl_visualize(os, CircuitGraphNode::getWidthForVisualize(), nQubits);
}

// implementation of countNumCircuitGraphs
namespace {
  void impl_getAllCircuitGraphs(IRNode* node,
                                std::vector<CircuitGraphNode*>& graphs);

  void impl_getAllCircuitGraphs(const CompoundNode& compoundNode,
                                std::vector<CircuitGraphNode*>& graphs) {
    for (const auto& node : compoundNode.nodes)
      impl_getAllCircuitGraphs(node.get(), graphs);
  }

  void impl_getAllCircuitGraphs(IRNode* node, 
                                std::vector<CircuitGraphNode*>& graphs) {
    if (auto* graphNode = llvm::dyn_cast<CircuitGraphNode>(node)) {
      graphs.push_back(graphNode);
      return ;
    }
    if (auto* ifNode = llvm::dyn_cast<IfMeasureNode>(node)) {
      impl_getAllCircuitGraphs(ifNode->thenBody, graphs);
      impl_getAllCircuitGraphs(ifNode->elseBody, graphs);
      return;
    }
    assert(false && "Unknown node type");
  }
}

std::vector<CircuitGraphNode*> CircuitNode::getAllCircuitGraphs() const {
  std::vector<CircuitGraphNode*> graphs;
  impl_getAllCircuitGraphs(body, graphs);
  return graphs;
}

unsigned CircuitNode::countNumCircuitGraphs() const {
  return getAllCircuitGraphs().size();
}

void CircuitNode::optimize(const FusionConfig& fusionConfig,
                           const CostModel* costModel) {
  auto allCircuitGraphs = getAllCircuitGraphs();
  for (auto* graph : allCircuitGraphs)
    applyGateFusion(fusionConfig, costModel, *graph);

}
