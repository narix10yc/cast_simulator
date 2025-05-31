#include "cast/Fusion.h"

#include "utils/iocolor.h"
#include "utils/PrintSpan.h"

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
int getKAfterFusion(QuantumGate* gateA, QuantumGate* gateB) {
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

/// @return Number of fused gates
int startFusion(ir::CircuitGraphNode& graph,
                const FusionConfig& config,
                const CostModel* costModel,
                const int maxK,
                row_iterator curIt,
                const int qubit) {
  
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
        os << graph.gateId(*item.gate)
           << "(nQubits=" << item.gate->nQubits()
           << ", opCount=" << item.gate->opCount(1e-8) << ")";
      });
    std::cerr << " => " << graph.gateId(*productGate)
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

} // anonymous namespace

void cast::applyGateFusion(
    const FusionConfig& config, const CostModel* costModel,
    ir::CircuitGraphNode& graph, int max_k) {
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
        nFused += startFusion(graph, config, costModel, curMaxK, it, q);
      ++it;
    }
    graph.squeeze();
  } while (nFused > 0 && ++curMaxK <= max_k);
}
