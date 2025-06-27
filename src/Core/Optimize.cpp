#include "cast/Core/Optimize.h"
#include "utils/iocolor.h"
#include "utils/PrintSpan.h"

#include "llvm/Support/Casting.h"

#define DEBUG_TYPE "fusion-new"
#include "llvm/Support/Debug.h"
// #define LLVM_DEBUG(X) X

using namespace cast;

const FusionConfig FusionConfig::Disable {
  .zeroTol = 0.0,
  .multiTraverse = false,
  .incrementScheme = false,
  .benefitMargin = 0.0,
};

const FusionConfig FusionConfig::Minor {
  .zeroTol = 1e-8,
  .multiTraverse = false,
  .incrementScheme = true,
  .benefitMargin = 0.5,
};

const FusionConfig FusionConfig::Default {
  .zeroTol = 1e-8,
  .multiTraverse = true,
  .incrementScheme = true,
  .benefitMargin = 0.2,
};

const FusionConfig FusionConfig::Aggressive {
  .zeroTol = 1e-8,
  .multiTraverse = true,
  .incrementScheme = true,
  .benefitMargin = 0.0,
};

namespace {

// Get the number of qubits after fusion of two quantum gates.
// We want this function because computing the matrix could be much more
// expensive.
int getKAfterFusion(const QuantumGate* gateA, const QuantumGate* gateB) {
  if (gateA == nullptr || gateB == nullptr)
    return 0; // no qubits
  int count = 0;
  auto itA = gateA->qubits().begin();
  auto itB = gateB->qubits().begin();
  const auto endA = gateA->qubits().end();
  const auto endB = gateB->qubits().end();
  while (itA != endA || itB != endB) {
    count++;
    if (itA == endA) {
      ++itB;
      continue;
    }
    if (itB == endB) {
      ++itA;
      continue;
    }
    if (*itA == *itB) {
      ++itA;
      ++itB;
      continue;
    }
    if (*itA > *itB) {
      ++itB;
      continue;
    }
    assert(*itA < *itB);
    ++itA;
    continue;
  }

  return count;
}

using row_iterator = ir::CircuitGraphNode::row_iterator;

struct TentativeFusedItem {
  // We must use shared pointers here because intermediate fusion steps may
  // remove gates from the graph, yet these gates may need to be restored.
  QuantumGatePtr gate;
  row_iterator iter;
};

} // end of anonymous namespace

/// @return Number of fused gates
int cast::startFusion(ir::CircuitGraphNode& graph,
                      const FusionConfig& config,
                      const CostModel* costModel,
                      int maxK,
                      row_iterator curIt,
                      int qubit) {
  
  // In the first stage of fusion, we greedily collect gates that may possibly
  // be fused together.
  // Later we will check if the fusion is beneficial. If not, we reject the
  // fusion and put back all these gates to their original positions.

  // productGate is the product of all fusedGates
  auto productGate = graph.lookup((*curIt)[qubit]);
  if (productGate == nullptr)
    return 0;
  auto fusedIt = curIt;

  // keep track of all gates that are being fused, and restore them upon
  // rejecting the fusion
  std::vector<TentativeFusedItem> fusedGates;
  fusedGates.reserve(8);
  fusedGates.emplace_back(productGate, fusedIt);

  // function that checks if candidateGate can be added to the fusedGates
  const auto checkFuseable = [&](QuantumGate* candidateGate) {
    if (candidateGate == nullptr)
      return false;
    // is candidateGate already in the fusedGates?
    for (const auto& [gate, row] : fusedGates) {
      if (gate.get() == candidateGate)
        return false;
    }
    return getKAfterFusion(productGate.get(), candidateGate) <= maxK;
  };

  // Start with same-row gates
  for (int q = qubit+1; q < graph.nQubits(); ++q) {
    auto* candidateGate = (*curIt)[q];
    if (productGate.get() == candidateGate)
      continue;
    if (checkFuseable(candidateGate) == false)
      continue;

    // candidateGate could be added to the fusedGates
    fusedGates.emplace_back(graph.lookup(candidateGate), curIt);
    graph.fuseAndInsertSameRow(curIt, q, productGate->qubits()[0]);
    productGate = graph.lookup((*curIt)[q]);
    assert(productGate != nullptr);
  }

  assert(curIt == fusedIt);

  // TODO: check logic here
  bool progress;
  do {
    curIt = std::next(fusedIt);
    if (curIt == graph.tile_end())
      break;

    progress = false;
    for (const auto& q : productGate->qubits()) {
      auto* candidateGate = (*curIt)[q];
      if (checkFuseable(candidateGate) == false)
        continue;
      // candidateGate is accepted
      fusedGates.emplace_back(graph.lookup(candidateGate), curIt);
      fusedIt = graph.fuseAndInsertDiffRow(std::prev(curIt), q);
      productGate = graph.lookup((*fusedIt)[candidateGate->qubits()[0]]);
      assert(productGate != nullptr);
      progress = true;
      break;
    }
  } while (progress == true);

  assert(fusedGates.size() > 0);
  if (fusedGates.size() == 1)
    return 0;

  assert(fusedIt != graph.tile_end());

  // Check benefit
  double oldTime = 0.0;
  for (const auto& [gate, iter] : fusedGates) {
    oldTime += costModel->computeGiBTime(
      gate, config.precision, config.nThreads);
  }
  double newTime = costModel->computeGiBTime(
    productGate, config.precision, config.nThreads);
  double benefit = oldTime / newTime - 1.0;
  LLVM_DEBUG(
    utils::printSpanWithPrinter(std::cerr,
      std::span(fusedGates.data(), fusedGates.size()),
      [&graph](std::ostream& os, const TentativeFusedItem& item) {
        os << graph.gateId(item.gate)
           << "(nQubits=" << item.gate->nQubits()
           << ", opCount=" << item.gate->opCount(1e-8) << ")";
      });
    std::cerr << " => " << graph.gateId(productGate)
              << "(nQubits=" << productGate->nQubits()
              << ", opCount=" << productGate->opCount(1e-8) << "). "
              << "Benefit = " << benefit << "; ";
  );

  // not enough benefit, undo this fusion
  if (benefit < config.benefitMargin) {
    LLVM_DEBUG(std::cerr << "Fusion Rejected\n");
    graph.removeGate(fusedIt, productGate->qubits()[0]);
    for (const auto& [gate, iter] : fusedGates)
      graph.insertGate(gate, iter);
    return 0;
  }
  // otherwise, enough benefit, accept this fusion
  LLVM_DEBUG(std::cerr << "Accepted\n";);
  // memory of fusedGates will be freed when this function returns
  return fusedGates.size() - 1;
}

void cast::applyGateFusion(ir::CircuitGraphNode& graph,
                           const FusionConfig& config,
                           const CostModel* costModel) {
  constexpr int MAX_K = 7;
  int curMaxK = 2;
  int nFused = 0;
  do {
    nFused = 0;
    auto it = graph.tile_begin();
    int q = 0;
    // we need to query graph.tile_end() every time, because startFusion may
    // change graph tile
    while (it != graph.tile_end()) {
      for (q = 0; q < graph.nQubits(); ++q) 
        nFused += cast::startFusion(graph, config, costModel, curMaxK, it, q);
      ++it;
    }
    graph.squeeze();
  } while (nFused > 0 && ++curMaxK <= MAX_K);
}

// helper functions to optimize
namespace {

using namespace cast::ir;

CircuitGraphNode*
getOrAppendCGNodeToCompoundNodeFront(CompoundNode& compoundNode) {
  if (!compoundNode.empty() && 
      llvm::isa<CircuitGraphNode>(compoundNode.nodes[0].get())) {
    return llvm::cast<CircuitGraphNode>(compoundNode.nodes[0].get());
  }
  compoundNode.push_front(std::make_unique<CircuitGraphNode>());
  return llvm::cast<CircuitGraphNode>(compoundNode.nodes[0].get());
}

CircuitGraphNode*
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

} // end of anynymous namespace

void cast::optimize(ir::CircuitNode& circuit,
                    const FusionConfig& config,
                    const CostModel* costModel) {
  const auto end = circuit.body.nodes.end();
  auto curIt = circuit.body.nodes.begin();
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
