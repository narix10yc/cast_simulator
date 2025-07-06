#include "cast/Core/Optimize.h"
#include "cast/Core/ImplOptimize.h"

#include "utils/PrintSpan.h"

#include "llvm/Support/Casting.h"

#define DEBUG_TYPE "impl-optimize"
#include "llvm/Support/Debug.h"
// #define LLVM_DEBUG(X) X

using namespace cast;

// helper functions for impl::startFusion
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
int cast::impl::startFusion(ir::CircuitGraphNode& graph,
                            const FusionConfig& config,
                            int max_k_candidate,
                            row_iterator curIt,
                            int qubit) {
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
    if (productGate.get() == candidateGate)
      return false;

    // is candidateGate already in the fusedGates?
    for (const auto& [gate, row] : fusedGates) {
      if (gate.get() == candidateGate)
        return false;
    }
    return getKAfterFusion(productGate.get(), candidateGate) <= max_k_candidate;
  };

  // Start with same-row gates
  for (int q = qubit+1; q < graph.nQubits(); ++q) {
    auto* candidateGate = (*curIt)[q];
    if (checkFuseable(candidateGate) == false)
      continue;

    // candidateGate could be added to the fusedGates
    fusedGates.emplace_back(graph.lookup(candidateGate), curIt);
    graph.fuseAndInsertSameRow(curIt, q, productGate->qubits()[0]);
    productGate = graph.lookup((*curIt)[q]);
    assert(productGate != nullptr);
  }

  assert(curIt == fusedIt);

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

  if (config.costModel == nullptr)
    return fusedGates.size() - 1;

  // Check benefit
  double oldTime = 0.0;
  for (const auto& [gate, iter] : fusedGates) {
    oldTime += config.costModel->computeGiBTime(
      gate, config.precision, config.nThreads);
  }
  double newTime = config.costModel->computeGiBTime(
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

int cast::impl::applySizeOnlyFusion(ir::CircuitGraphNode& graph,
                                    int max_k,
                                    double swaTol) {
  int nFused = 0;
  auto it = graph.tile_begin();
  while (it != graph.tile_end()) {
    for (int q = 0; q < graph.nQubits(); ++q) {
      nFused += impl::startFusion(
        graph, FusionConfig::SizeOnly(max_k), max_k, it, q);
    }
    ++it;
  }
  graph.squeeze();
  return nFused;
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

// This class is used to keep track of which qubits have been checked.
// It saves memory by using a single 64-bit integer with bitwise operations.
// Therefore, it can only handle up to 64 qubits.
class FlagArray {
  uint64_t flags;
public:
  // initialize: [0, nQubits) bits are set to 0 (not checked)
  // other bits are set to 1 (checked).
  FlagArray(int nQubits) : flags(~((1ULL << nQubits) - 1)) {
    assert(nQubits < 64 && "nQubits should be less than 64");
  }
  
  void setCheck(int q) {
    assert(q >= 0 && q < 64 && "q should be in [0, 64)");
    flags |= (1ULL << q);
  }
  
  void setNotCheck(int q) {
    assert(q >= 0 && q < 64 && "q should be in [0, 64)");
    flags &= ~(1ULL << q);
  }  

  bool allChecked() const {
    return flags == ~(static_cast<uint64_t>(0));
  }

  // Possibly return -1 if all qubits are checked.
  int getFirstUnchecked() const {
    if (allChecked())
      return -1; // all qubits are checked
    for (int q = 0; q < 64; ++q) {
      if (((flags >> q) & 1) == 0ULL)
        return q; // return the first unchecked qubit
    }
    assert(false && "Unreachable: all qubits should be checked");
    return -1; // should never reach here
  }

  bool isChecked(int q) const {
    assert(q >= 0 && q < 64 && "q should be in [0, 64)");
    return (flags >> q) & 1;
  }

  bool isNotChecked(int q) const {
    assert(q >= 0 && q < 64 && "q should be in [0, 64)");
    return !isChecked(q);
  }

  bool operator[](int q) const {
    return isChecked(q);
  }
};

// Check if the gate commutes with the measurement on measureQubit.
// This function currently does not consider all cases. If it returns true,
// then commutation is guaranteed. The converse is not true.
// The tolerance must be set to a positive value. Users are advised to check 
// the tolerance before calling this function (for example, to disable
// swapping analysis).
bool isCommutingWithMeasurement(const QuantumGate* gate,
                                int measureQubit,
                                double tol) {
  assert(gate != nullptr);
  assert(tol > 0.0 && "Tolerance should be positive");
  assert(measureQubit >= 0);
  const auto& qubits = gate->qubits();
  if (std::ranges::find(qubits, measureQubit) == qubits.end())
    return true;

  switch (gate->nQubits()) {
    case 0:
      // this should never be reached
      assert(false && "Gate targets is empty");
      return true;
    case 1:
      // unless gate is the identity
      return false;
    case 2: {
      auto stdQuGate = llvm::dyn_cast<StandardQuantumGate>(gate);
      if (stdQuGate == nullptr)
        return false; // cannot determine commutativity; not supported yet
      auto scalarGM = stdQuGate->getScalarGM();
      if (scalarGM == nullptr)
        return false; // cannot determine commutativity; not supported yet
      const auto& m = scalarGM->matrix();
      if (measureQubit == qubits[0]) {
        // is everything all close to zero?
        bool b = m.real(0, 1) < tol && m.real(0, 3) < tol &&
                 m.real(1, 0) < tol && m.real(1, 2) < tol &&
                 m.real(2, 1) < tol && m.real(2, 3) < tol &&
                 m.real(3, 0) < tol && m.real(3, 2) < tol && 
                 m.imag(0, 1) < tol && m.imag(0, 3) < tol &&
                 m.imag(1, 0) < tol && m.imag(1, 2) < tol &&
                 m.imag(2, 1) < tol && m.imag(2, 3) < tol &&
                 m.imag(3, 0) < tol && m.imag(3, 2) < tol;
        return b;
      } 
      // else
      assert(measureQubit == qubits[1]);
      bool b = m.real(0, 2) < tol && m.real(0, 3) < tol &&
               m.real(1, 2) < tol && m.real(1, 3) < tol &&
               m.real(2, 0) < tol && m.real(2, 1) < tol &&
               m.real(3, 0) < tol && m.real(3, 1) < tol &&
               m.imag(0, 2) < tol && m.imag(0, 3) < tol &&
               m.imag(1, 2) < tol && m.imag(1, 3) < tol &&
               m.imag(2, 0) < tol && m.imag(2, 1) < tol &&
               m.imag(3, 0) < tol && m.imag(3, 1) < tol;
      return b;
    }
    default:
      // We could use cast::isCommuting() to check if gate commutes with |0><0|
      // and |1><1|
      return false;
  }
}

// returns true if the pass takes effect
// move single-qubit gates not on the measurement wire at the bottom of the 
// prior circuit graph node to the top of the if node
bool applyPriorIfPass(CircuitGraphNode* priorCGNode,
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
bool applyIfJoinPass(IfMeasureNode* ifNode,
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

// Return a vector of gates, in order, from the bottom of the tile that commutes
// with the measurement on \c measureQubit.
// To put back the gates, reversely iterate through the returned vector.
// swaTol: swapping analysis tolerance. Set to a non-positive value to disable
// swapping analysis.
std::vector<QuantumGatePtr> collectCommutingGatesWithMeasurement(
    CircuitGraphNode& graph, int measureQubit, double swaTol) {
  std::vector<QuantumGatePtr> gates;
  if (graph.tile().empty())
    return gates;
  auto begin = graph.tile_begin();
  auto it = graph.tile_end();
  const auto nQubits = graph.nQubits();
  FlagArray flags(nQubits);
  while (it != begin) {
    --it;
    // find the first non-checked qubit
    int q = flags.getFirstUnchecked();
    if (q < 0) {
      // all qubits are checked, we are done
      break;
    }

    auto gate = graph.lookup((*it)[q]);
    if (gate == nullptr)
      continue;

    bool accept = true;
    switch (gate->nQubits()) {
      case 1: {
        if (q == measureQubit) {
          // swaTol <= 0.0 means we disable the swapping analysis
          if (swaTol <= 0.0 ||
              !isCommutingWithMeasurement(gate.get(), measureQubit, swaTol)) {
            accept = false;
            break;
          }
          accept = true;
          break;
        }
        // else: q is not the measurement qubit
        accept = true;
        break;
      }
      case 2: {
        int q0 = gate->qubits()[0];
        int q1 = gate->qubits()[1];
        if (q0 != measureQubit && q1 != measureQubit) {
          accept = true;
          break;
        }
        // else: one of the qubits is the measurement qubit
        if (swaTol <= 0.0 ||
            !isCommutingWithMeasurement(gate.get(), measureQubit, swaTol)) {
          accept = false;
          break;
        }
        accept = true;
        break;
      }
      default: {
        assert(false && "Unreachable");
        break;
      }
    } // end of switch

    if (accept) {
      gates.push_back(gate);
      graph.removeGate(it, q);
    } else {
      // gate is not accepted. Mark the corresponding wires as checked (i.e.
      // disallow further gates to commute)
      switch (gate->nQubits()) {
        case 1:
          flags.setCheck(q);
          break;
        case 2: {
          flags.setCheck(gate->qubits()[0]);
          flags.setCheck(gate->qubits()[1]);
          break;
        }
        default: {
          assert(false && "Unreachable");
          break;
        }
      } // end of switch
    } // end of if-else

  } // end of while(it != begin)

}

bool applyFusionCFOPass_PriorIf(CircuitGraphNode* priorCGNode,
                                IfMeasureNode* ifNode, 
                                const FusionConfig& config) {
  assert(priorCGNode != nullptr);
  assert(ifNode != nullptr);

  if (priorCGNode->tile().empty())
    return false; // nothing to do

  bool hasEffect = false;
  auto thenCGNode = getOrAppendCGNodeToCompoundNodeBack(ifNode->thenBody);
  auto elseCGNode = getOrAppendCGNodeToCompoundNodeBack(ifNode->elseBody);

  const int nQubits = priorCGNode->nQubits();
  assert(nQubits < 64);
  FlagArray thenMask(nQubits);
  FlagArray elseMask(nQubits);

  while (!thenMask.allChecked() || !elseMask.allChecked()) {
    // find the first qubit that is not checked
    int q;
    for (q = 0; q < nQubits; ++q) {
      if (thenMask.isNotChecked(q))
        break;
      if (elseMask.isNotChecked(q))
        break;
    }
    assert(q < nQubits && "There should be at least one qubit left to check");

    // Move the gate from the join block to the if block
    auto priorRow = priorCGNode->tile_end();
    --priorRow; // the last row
    auto gate = priorCGNode->lookup((*priorRow)[q]);
    if (gate == nullptr) {
      // remove the qubit from the mask
      thenMask.setCheck(q);
      elseMask.setCheck(q);
      continue;
    }
    auto thenRowIt = thenCGNode->insertGate(gate, thenCGNode->tile_begin());
    auto elseRowIt = elseCGNode->insertGate(gate, elseCGNode->tile_begin());
    // We will squeeze anyway. insertGate potentially creates bubbles
    // thenCGNode->squeeze();
    // elseCGNode->squeeze();
    priorCGNode->removeGate(priorRow, q);

    auto thenFused = cast::impl::startFusion(
      *thenCGNode, config, max_k, thenRowIt, q);
    auto elseFused = cast::impl::startFusion(
      *elseCGNode, config, max_k, elseRowIt, q);
    if (thenFused > 0 || elseFused > 0) {
      hasEffect = true;
      // We do not need to squeeze the priorCGNode because gates are removed
      // from the end of it
    } else {
      // no effect
      // put the gate back to the join block
      thenCGNode->removeGate(thenRowIt, q);
      elseCGNode->removeGate(elseRowIt, q);
      priorCGNode->insertGate(gate, priorRow);
      // TODO: we could check if thenCGNode and elseCGNode need to be squeezed
      // by checking the which row 
      thenCGNode->squeeze();
      elseCGNode->squeeze();
      // remove the qubit from the mask
      thenMask.setCheck(q);
      elseMask.setCheck(q);
      continue;
    }
  }
  return hasEffect;
}

bool applyFusionCFOPass_IfJoin(IfMeasureNode* ifNode, 
                               CircuitGraphNode* joinCGNode,
                               const FusionConfig& config,
                               const CostModel* costModel,
                               int max_k) {
  assert(joinCGNode != nullptr);
  assert(ifNode != nullptr);
  bool hasEffect = false;
  auto thenCGNode = getOrAppendCGNodeToCompoundNodeBack(ifNode->thenBody);
  auto elseCGNode = getOrAppendCGNodeToCompoundNodeBack(ifNode->elseBody);

  const int nQubits = joinCGNode->nQubits();
  assert(nQubits < 64);
  FlagArray thenMask(nQubits);
  FlagArray elseMask(nQubits);

  // TODO: check commutation with measurement
  thenMask.setCheck(ifNode->qubit);
  elseMask.setCheck(ifNode->qubit);

  while (!thenMask.allChecked() || !elseMask.allChecked()) {
    // find the first qubit that is not checked
    int q;
    for (q = 0; q < nQubits; ++q) {
      if (thenMask.isNotChecked(q))
        break;
      if (elseMask.isNotChecked(q))
        break;
    }
    assert(q < nQubits && "There should be at least one qubit left to check");

    // Move the gate from the join block to the if block
    auto gate = joinCGNode->lookup((*joinCGNode->tile_begin())[q]);
    // TODO: check eligibility 
    if (gate == nullptr) {
      // remove the qubit from the mask
      thenMask.setCheck(q);
      elseMask.setCheck(q);
      continue;
    }
    auto thenRowIt = thenCGNode->insertGate(gate, thenCGNode->tile_end());
    auto elseRowIt = elseCGNode->insertGate(gate, elseCGNode->tile_end());
    joinCGNode->removeGate(joinCGNode->tile_begin(), q);
    int thenFused = 0, elseFused = 0;
    if (thenRowIt != thenCGNode->tile_begin()) {
      thenFused = cast::impl::startFusion(
        *thenCGNode, config, max_k, std::prev(thenRowIt), q);
    }
    if (elseRowIt != elseCGNode->tile_begin()) {
      elseFused = cast::impl::startFusion(
        *elseCGNode, config, max_k, std::prev(elseRowIt), q);
    }
    if (thenFused > 0 || elseFused > 0) {
      hasEffect = true;
      joinCGNode->squeeze();
      if (thenFused > 0 && thenRowIt != thenCGNode->tile_begin())
        thenCGNode->squeeze(std::prev(thenRowIt));
      if (elseFused > 0 && elseRowIt != elseCGNode->tile_begin())
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
      thenMask.setCheck(q);
      elseMask.setCheck(q);
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

} // end of anonymous namespace

bool cast::impl::applyFusionCFOPass(ir::CircuitNode& circuit,
                                    const FusionConfig& config) {
  bool hasEffect = false;
  const auto end = circuit.body.nodes.end();
  auto curIt = circuit.body.nodes.begin();
  while (curIt != end) {
    ++curIt;
    if (curIt == end)
      break;
    auto prevIt = std::prev(curIt);
    assert(prevIt != end && "prevIt should never be end");

    if (auto* priorCGNode = llvm::dyn_cast<CircuitGraphNode>(prevIt->get())) {
      if (auto* ifNode = llvm::dyn_cast<IfMeasureNode>(curIt->get())) {
        hasEffect |= applyFusionCFOPass_PriorIf(priorCGNode, ifNode, config);
      }
    }
    else if (auto* ifNode = llvm::dyn_cast<IfMeasureNode>(prevIt->get())) {
      if (auto* joinCGNode = llvm::dyn_cast<CircuitGraphNode>(curIt->get())) {
        hasEffect |= applyFusionCFOPass_IfJoin(ifNode, joinCGNode, config);
      }
    }
  } // end of while loop
  return hasEffect;
}

#undef DEBUG_TYPE