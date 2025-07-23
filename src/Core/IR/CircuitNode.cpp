#include "cast/Core/IRNode.h"
#include "utils/iocolor.h"
#include "llvm/Support/Casting.h"

using namespace cast::ir;

std::ostream& CircuitNode::print(std::ostream& os, int indent) const {
  writeIndent(os, indent) << "cast.circuit(" << name << ") {\n";
  body.print(os, indent + 1);
  writeIndent(os, indent) << "}\n";
  return os;
}

// implementation of num gates in critical path
namespace {
int impl_nGatesCriticalPath(const IRNode* node);

int impl_nGatesCriticalPath(const CompoundNode* compoundNode) {
  int length = 0;
  for (const auto& child : compoundNode->nodes) {
    length += impl_nGatesCriticalPath(child.get());
  }
  return length;
}

int impl_nGatesCriticalPath(const CircuitNode* circuitNode) {
  return impl_nGatesCriticalPath(&circuitNode->body);
}

int impl_nGatesCriticalPath(const CircuitGraphNode* graphNode) {
  return graphNode->nGates();
}

int impl_nGatesCriticalPath(const IfMeasureNode* ifNode) {
  int thenLength = impl_nGatesCriticalPath(&ifNode->thenBody);
  int elseLength = impl_nGatesCriticalPath(&ifNode->elseBody);
  return std::max(thenLength, elseLength);
}

int impl_nGatesCriticalPath(const IRNode* node) {
  if (auto* n = llvm::dyn_cast<CompoundNode>(node))
    return impl_nGatesCriticalPath(n);
  if (auto* n = llvm::dyn_cast<CircuitNode>(node))
    return impl_nGatesCriticalPath(n);
  if (auto* n = llvm::dyn_cast<CircuitGraphNode>(node))
    return impl_nGatesCriticalPath(n);
  if (auto* n = llvm::dyn_cast<IfMeasureNode>(node))
    return impl_nGatesCriticalPath(n);
  assert(false && "Unknown node type in impl_nGatesCriticalPath");
  return 0; // unreachable
}

double impl_opcount(const IRNode* node, double zeroTol);

double impl_opcount(const CompoundNode* compoundNode, double zeroTol) {
  double count = 0.0;
  for (const auto& child : compoundNode->nodes) {
    count += impl_opcount(child.get(), zeroTol);
  }
  return count;
}

double impl_opcount(const CircuitNode* circuitNode, double zeroTol) {
  return impl_opcount(&circuitNode->body, zeroTol);
}

double impl_opcount(const CircuitGraphNode* graphNode, double zeroTol) {
  double count = 0.0;
  auto gates = graphNode->getAllGates();
  for (const auto& gate : gates) {
    count += gate->opCount(zeroTol); // Pass zeroTol to opCount
  }
  return count;
}

double impl_opcount(const IfMeasureNode* ifNode, double zeroTol) {
  double thenCount = impl_opcount(&ifNode->thenBody, zeroTol);
  double elseCount = impl_opcount(&ifNode->elseBody, zeroTol);
  return std::max(thenCount, elseCount);
}

double impl_opcount(const IRNode* node, double zeroTol) {
  if (auto* n = llvm::dyn_cast<CompoundNode>(node))
    return impl_opcount(n, zeroTol);
  if (auto* n = llvm::dyn_cast<CircuitNode>(node))
    return impl_opcount(n, zeroTol);
  if (auto* n = llvm::dyn_cast<CircuitGraphNode>(node))
    return impl_opcount(n, zeroTol);
  if (auto* n = llvm::dyn_cast<IfMeasureNode>(node))
    return impl_opcount(n, zeroTol);
  assert(false && "Unknown node type in impl_opcount");
  return 0; // unreachable
}

} // end of anonymous namespace

std::ostream& CircuitNode::displayInfo(std::ostream& os, int verbose) const {
  os << BOLDCYAN("=== Info of ir::CircuitNode @ " << this << " === ")
     << "(Verbose " << verbose << ")\n";

  os << CYAN("- Name: ") << name << "\n";
  auto graphs = getAllCircuitGraphs();
  os << CYAN("- Num CircuitGraphs: ") << graphs.size() << "\n";
  int nGates = 0;
  for (const auto& graph : graphs)
    nGates += graph->nGates();
  os << CYAN("- Num Gates: ") << nGates << "\n"
     << "             " << impl_nGatesCriticalPath(&body)
     << " in critical path\n";
  os << CYAN("- Op Count:  ") << impl_opcount(&body, 1e-8)
     << " in critical path\n"
     << CYAN("             ") << impl_opcount(&body, 0.0) << " if dense\n";

  os << BOLDCYAN("====================================") << "\n";
  return os;
}

std::ostream&
CircuitNode::impl_visualize(std::ostream& os, int width, int n_qubits) const {
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
    return;
  }
  if (auto* ifNode = llvm::dyn_cast<IfMeasureNode>(node)) {
    impl_getAllCircuitGraphs(ifNode->thenBody, graphs);
    impl_getAllCircuitGraphs(ifNode->elseBody, graphs);
    return;
  }
  assert(false && "Unknown node type");
}
} // namespace

std::vector<CircuitGraphNode*> CircuitNode::getAllCircuitGraphs() const {
  std::vector<CircuitGraphNode*> graphs;
  impl_getAllCircuitGraphs(body, graphs);
  return graphs;
}

unsigned CircuitNode::countNumCircuitGraphs() const {
  return getAllCircuitGraphs().size();
}
