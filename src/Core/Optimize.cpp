#include "cast/Core/Optimize.h"
#include "utils/iocolor.h"
#include "utils/PrintSpan.h"

#include "llvm/Support/Casting.h"

#define DEBUG_TYPE "fusion-new"
#include "llvm/Support/Debug.h"
// #define LLVM_DEBUG(X) X

using namespace cast;

const FusionConfig FusionConfig::Minor {
  .zeroTol = 1e-8,
  .incrementScheme = true,
  .maxKOverride = 3,
  .benefitMargin = 0.2,
};

const FusionConfig FusionConfig::Default {
  .zeroTol = 1e-8,
  .incrementScheme = true,
  .maxKOverride=6,
  .benefitMargin = 0.0,
};

const FusionConfig FusionConfig::Aggressive {
  .zeroTol = 1e-8,
  .incrementScheme = true,
  .maxKOverride = GLOBAL_MAX_K,
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
                      int cur_max_k,
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
    return getKAfterFusion(productGate.get(), candidateGate) <= cur_max_k;
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
  int curMaxK = (config.incrementScheme ? 2 : config.maxKOverride);
  do {
    auto it = graph.tile_begin();
    // we need to query graph.tile_end() every time, because startFusion may
    // change graph tile
    while (it != graph.tile_end()) {
      for (int q = 0; q < graph.nQubits(); ++q) 
        cast::startFusion(graph, config, costModel, curMaxK, it, q);
      ++it;
    }
    graph.squeeze();
  } while (++curMaxK <= config.maxKOverride);
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
        break;
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

bool applyIfJoinFusionPass(IfMeasureNode* ifNode, 
                           CircuitGraphNode* joinCGNode,
                           const FusionConfig& config,
                           const CostModel* costModel) {
  assert(joinCGNode != nullptr);
  assert(ifNode != nullptr);
  bool hasEffect = false;
  auto thenCGNode = getOrAppendCGNodeToCompoundNodeBack(ifNode->thenBody);
  auto elseCGNode = getOrAppendCGNodeToCompoundNodeBack(ifNode->elseBody);

  const int nQubits = joinCGNode->nQubits();
  assert(nQubits < 64);
  uint64_t thenMask = (1ULL << nQubits) - 1;
  uint64_t elseMask = (1ULL << nQubits) - 1;

  while (thenMask != 0 || elseMask != 0) {
    // find the first qubit that is not checked
    int q;
    for (q = 0; q < nQubits; ++q) {
      if ((thenMask >> q) & 1)
        break;
      if ((elseMask >> q) & 1)
        break;
    }
    assert(q < nQubits && "There should be at least one qubit left to check");

    // Move the gate from the join block to the if block
    auto gate = joinCGNode->lookup((*joinCGNode->tile_begin())[q]);
    if (gate == nullptr) {
      // remove the qubit from the mask
      thenMask &= ~(1ULL << q);
      elseMask &= ~(1ULL << q);
      continue;
    }
    auto thenRowIt = thenCGNode->insertGate(gate, thenCGNode->tile_end());
    auto elseRowIt = elseCGNode->insertGate(gate, elseCGNode->tile_end());
    joinCGNode->removeGate(joinCGNode->tile_begin(), q);
    int thenFused = 0, elseFused = 0;
    if (thenRowIt != thenCGNode->tile_begin()) {
      thenFused = cast::startFusion(
        *thenCGNode, config, costModel, 2, std::prev(thenRowIt), q);
    }
    if (elseRowIt != elseCGNode->tile_begin()) {
      elseFused = cast::startFusion(
        *elseCGNode, config, costModel, 2, std::prev(elseRowIt), q);
    }
    if (thenFused > 0 || elseFused > 0) {
      hasEffect = true;
      joinCGNode->squeeze();
      if (thenFused > 0)
        thenCGNode->squeeze(std::prev(thenRowIt));
      if (elseFused > 0)
        elseCGNode->squeeze(std::prev(elseRowIt));
    } else {
      // no effect
      // put the gate back to the join block
      thenCGNode->removeGate(thenRowIt, q);
      elseCGNode->removeGate(elseRowIt, q);
      joinCGNode->insertGate(gate, joinCGNode->tile_begin());
      thenCGNode->squeeze(thenRowIt);
      elseCGNode->squeeze(elseRowIt);
      // remove the qubit from the mask
      thenMask &= ~(1ULL << q);
      elseMask &= ~(1ULL << q);
      continue;
    }
  }
  return hasEffect;
}

bool applyPreFusionCFOStage1(ir::CircuitNode& circuit) {
  bool hasEffect = false;
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

    if (auto* ifNode = llvm::dyn_cast<IfMeasureNode>(curIt->get())) {
      if (auto* priorCGNode = llvm::dyn_cast<CircuitGraphNode>(prevIt->get()))
        hasEffect |= applyPriorIfPass(priorCGNode, ifNode);
      if (auto* joinCGNode = llvm::dyn_cast<CircuitGraphNode>(nextIt->get()))
        hasEffect |= applyIfJoinPass(ifNode, joinCGNode);
    }
  }
  return hasEffect;
}

bool applyPreFusionCFOStage3(ir::CircuitNode& circuit,
                             const FusionConfig& config,
                             const CostModel* costModel) {
  bool hasEffect = false;
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

    if (auto* ifNode = llvm::dyn_cast<IfMeasureNode>(curIt->get())) {
      if (auto* joinCGNode = llvm::dyn_cast<CircuitGraphNode>(nextIt->get()))
        hasEffect |= applyIfJoinFusionPass(
          ifNode, joinCGNode, config, costModel);
    }
  }
  return hasEffect;
}

} // end of anynymous namespace

void cast::applyGateFusion(ir::CircuitNode& circuit,
                           const FusionConfig& config,
                           const CostModel* costModel) {
  // perform fusion in each block
  auto allCircuitGraphs = circuit.getAllCircuitGraphs();
  for (auto* graph : allCircuitGraphs)
    applyGateFusion(*graph, config, costModel);

}

void cast::applyPreFusionCFOPass(ir::CircuitNode& circuit) {
  applyPreFusionCFOStage1(circuit);

  // perform fusion in each block
  auto fusionConfig = FusionConfig::Aggressive;
  fusionConfig.maxKOverride = 2; // size-only fusion with k=2
  NaiveCostModel costModel(2, -1, 1e-8);
  auto allCircuitGraphs = circuit.getAllCircuitGraphs();
  for (auto* graph : allCircuitGraphs)
    applyGateFusion(*graph, fusionConfig, &costModel);

  applyPreFusionCFOStage3(circuit, fusionConfig, &costModel);
}