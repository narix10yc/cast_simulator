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
  auto graphs = getAllCircuitGraphs();
  os << CYAN("- Num CircuitGraph: ") << graphs.size() << "\n";
  int nGates = 0;
  for (const auto& graph : graphs)
    nGates += graph->nGates();
  os << CYAN("- Num Gates: ") << nGates << "\n";

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


static CircuitGraphNode*
getOrAppendCGNodeToCompoundNodeFront(CompoundNode& compoundNode) {
  if (!compoundNode.empty() && 
      llvm::isa<CircuitGraphNode>(compoundNode.nodes[0].get())) {
    return llvm::cast<CircuitGraphNode>(compoundNode.nodes[0].get());
  }
  compoundNode.push_front(std::make_unique<CircuitGraphNode>());
  return llvm::cast<CircuitGraphNode>(compoundNode.nodes[0].get());
}

static CircuitGraphNode*
getOrAppendCGNodeToCompoundNodeBack(CompoundNode& compoundNode) {
  if (!compoundNode.empty() && 
      llvm::isa<CircuitGraphNode>(compoundNode.nodes.back().get())) {
    return llvm::cast<CircuitGraphNode>(compoundNode.nodes.back().get());
  }
  compoundNode.push_back(std::make_unique<CircuitGraphNode>());
  return llvm::cast<CircuitGraphNode>(compoundNode.nodes.back().get());
}

// returns true if the pass takes effect
// move single-qubit gates not on the measurement wire at the bottom of the 
// prior circuit graph node to the top of the if node
static bool applyPriorIfPass(CircuitGraphNode* priorCGNode,
                             IfMeasureNode* ifNode) {
  assert(priorCGNode != nullptr);
  assert(ifNode != nullptr);
  bool hasEffect = false;
  auto thenCGNode = getOrAppendCGNodeToCompoundNodeFront(ifNode->thenBody);
  auto elseCGNode = getOrAppendCGNodeToCompoundNodeFront(ifNode->elseBody);
  
  const int nQubits = priorCGNode->nQubits();
  const auto begin = priorCGNode->tile_begin();
  for (int q = 0; q < nQubits; ++q) {
    if (q == ifNode->qubit)
      continue; // skip the measurement wire
    auto it = priorCGNode->tile_end();
    while (it != begin) {
      --it;
      auto gate = priorCGNode->lookup((*it)[q]);
      if (gate == nullptr)
        continue;
      if (gate->nQubits() != 1)
        continue;
      thenCGNode->insertGate(gate, thenCGNode->tile_begin());
      elseCGNode->insertGate(gate, elseCGNode->tile_begin());
      priorCGNode->removeGate(it, q);
      hasEffect = true;
    }
  }
  if (hasEffect) {
    thenCGNode->squeeze();
    elseCGNode->squeeze();
  }

  return hasEffect;
}

// returns true if the pass takes effect
// move all single-qubit gates at the top of the join circuit graph node to 
// the bottom of the if node
static bool applyIfJoinPass(IfMeasureNode* ifNode,
                            CircuitGraphNode* joinCGNode) {
  assert(joinCGNode != nullptr);
  assert(ifNode != nullptr);
  bool hasEffect = false;
  auto thenCGNode = getOrAppendCGNodeToCompoundNodeBack(ifNode->thenBody);
  auto elseCGNode = getOrAppendCGNodeToCompoundNodeBack(ifNode->elseBody);

  const int nQubits = joinCGNode->nQubits();
  auto end = joinCGNode->tile_end();
  for (int q = 0; q < nQubits; ++q) {
    auto it = joinCGNode->tile_begin();
    do {
      auto gate = joinCGNode->lookup((*it)[q]);
      if (gate == nullptr)
        continue;
      if (gate->nQubits() != 1)
        continue;
      thenCGNode->insertGate(gate, thenCGNode->tile_end());
      elseCGNode->insertGate(gate, elseCGNode->tile_end());
      joinCGNode->removeGate(it, q);
      hasEffect = true;
    } while (++it != end);
  }
  if (hasEffect) {
    thenCGNode->squeeze();
    elseCGNode->squeeze();
  }

  return hasEffect;
}

void CircuitNode::optimize(const FusionConfig& fusionConfig,
                           const CostModel* costModel) {
  const auto end = body.nodes.end();
  auto curIt = body.nodes.begin();
  while (curIt != end) {
    ++curIt;
    if (curIt == end)
      break;
    auto prevIt = std::prev(curIt);
    auto nextIt = std::next(curIt);
    assert(prevIt != end && "prevIt should never be end");
    if (nextIt == end)
      break;

    bool hasEffect = false;
    if (auto* ifNode = llvm::dyn_cast<IfMeasureNode>(curIt->get())) {
      if (auto* priorCGNode = llvm::dyn_cast<CircuitGraphNode>(prevIt->get()))
        hasEffect |= applyPriorIfPass(priorCGNode, ifNode);
      if (auto* joinCGNode = llvm::dyn_cast<CircuitGraphNode>(nextIt->get()))
        hasEffect |= applyIfJoinPass(ifNode, joinCGNode);
    }
  }

  // perform fusion in each block
  // auto allCircuitGraphs = getAllCircuitGraphs();
  // for (auto* graph : allCircuitGraphs)
    // applyGateFusion(fusionConfig, costModel, *graph);

}
