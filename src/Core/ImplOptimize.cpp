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

int absorbSingleQubitGate(ir::CircuitGraphNode& graph,
                          row_iterator it,
                          int qubit) {
  assert(it != graph.tile_end());
  auto* gate = (*it)[qubit];
  assert(gate != nullptr && gate->nQubits() == 1 && 
        "We expect a single-qubit gate when calling absorbSingleQubitGate");

  // Try to fuse it with gates in next rows
  auto end = graph.tile_end();
  auto curIt = it;
  while (++curIt != end) {
    auto* nextGate = (*curIt)[qubit];
    if (nextGate == nullptr)
      continue;
    
    // fuse them together
    auto fusedGate = cast::matmul(nextGate, gate);
    assert(fusedGate != nullptr);
    graph.removeGate(it, qubit);
    graph.removeGate(curIt, qubit);
    auto inserted = graph.insertGate(fusedGate, curIt);
    assert(inserted == curIt &&
           "curIt should have enough space for the fused gate");
    return 1; // fused one gate
  }

  // Try to fuse it with gates in previous rows
  auto begin = graph.tile_begin();
  curIt = it;
  while (curIt != begin) {
    --curIt;
    auto* prevGate = (*curIt)[qubit];
    if (prevGate == nullptr)
      continue;

    // fuse them together
    auto fusedGate = cast::matmul(gate, prevGate);
    assert(fusedGate != nullptr);
    graph.removeGate(it, qubit);
    graph.removeGate(curIt, qubit);
    auto inserted = graph.insertGate(fusedGate, curIt);
    assert(inserted == curIt &&
           "curIt should have enough space for the fused gate");
    return 1; // fused one gate
  }

  return 0; // no fusion happened
}

// We expect gateA, gateG, and gateH are on three consecutive rows with at least
// one shared target qubit. This function checks if it might be beneficial to
// swap gateG and gateH, so that we can fuse gateA with the gateH. That is, 
// instead of doing HGA, we do GHA, with HA fused together.
// When gateG and gateH are already very large, checking their commutation
// may be super expensive. We skip the check if the gateG * gateH would act on
// more than max_k_cutoff qubits.
bool checkSwappable(const QuantumGate* gateA,
                    const QuantumGate* gateG,
                    const QuantumGate* gateH,
                    int max_k_candidate,
                    int max_k_cutoff,
                    double swaTol) {
  if (swaTol <= 0.0)
    return false; // swapping is disabled
  
  if (gateA == nullptr || gateG == nullptr || gateH == nullptr)
    return false; // no gates to swap

  if (getKAfterFusion(gateA, gateH) > max_k_candidate)
    return false; // resulting gate would be too large
  
  if (getKAfterFusion(gateG, gateH) > max_k_cutoff)
    return false; // commutation check is too expensive
  
  return cast::isCommuting(gateG, gateH, swaTol);
}

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
  int nFused = 0;

  // keep track of all gates that are being fused, and restore them upon
  // rejecting the fusion
  std::vector<TentativeFusedItem> fusedGates;
  fusedGates.reserve(4);
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
  for (int q = qubit + 1; q < graph.nQubits(); ++q) {
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
      if (checkFuseable(candidateGate) == false) {
        if (config.swaTol <= 0.0)
          continue; // swapping is disabled
        // check if we can swap candidateGate with the next gate
        auto nextIt = std::next(curIt);
        if (nextIt == graph.tile_end())
          continue; // no next row to swap with
        auto* nextGate = (*nextIt)[q];
        if (checkSwappable(productGate.get(), candidateGate, nextGate,
                           max_k_candidate,
                           GLOBAL_MAX_K + 1, // max_k_cutoff
                           config.swaTol) == false) {
          // std::cerr << "Not triggering swap\n";
          continue;
        }
        // We can swap candidateGate with nextGate
        LLVM_DEBUG(
          std::cerr << "Swapping gates at (" 
                    << graph.gateId(productGate) << ", " 
                    << graph.gateId(candidateGate) << ") with gate "
                    << graph.gateId(nextGate) << "\n";
        );
        graph.swapGates(curIt, q);
        graph.squeeze(nextIt);
        // swapping may add a new row between fusedIt and curIt
        curIt = std::next(fusedIt);
        assert((*curIt)[q] == nextGate);
        candidateGate = nextGate;
      }
      // candidateGate is accepted
      fusedGates.emplace_back(graph.lookup(candidateGate), curIt);
      assert(curIt == std::next(fusedIt));
      fusedIt = graph.fuseAndInsertDiffRow(fusedIt, q);
      productGate = graph.lookup((*fusedIt)[candidateGate->qubits()[0]]);
      assert(productGate != nullptr);
      progress = true;
      break;
    }
  } while (progress == true);

  assert(fusedGates.size() > 0);
  if (fusedGates.size() == 1)
    return nFused;

  assert(fusedIt != graph.tile_end());

  if (config.costModel == nullptr)
    return nFused + fusedGates.size() - 1;

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
    return nFused;
  }
  // otherwise, enough benefit, accept this fusion
  LLVM_DEBUG(std::cerr << "Accepted\n";);
  // memory of fusedGates will be freed when this function returns
  return nFused + fusedGates.size() - 1;
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

int cast::impl::applySizeTwoFusion(ir::CircuitGraphNode& graph, double swaTol) {
  // Step 1: absorb single-qubit gates
  // When step 1 finishes, all single-qubit gates should be absorbed. The only
  // exception is when there is precisely one single-qubit gate in a qubit wire.
  // In that case, we will wait for the main fusion algorithm to handle it.
  auto it = graph.tile_begin();
  int nFused = 0;
  do {
    for (int q = 0; q < graph.nQubits(); ++q) {
      auto* gate = (*it)[q];
      if (gate == nullptr || gate->nQubits() != 1)
        continue;
      nFused += absorbSingleQubitGate(graph, it, q);
    }
  } while (++it != graph.tile_end());
  graph.squeeze();

  // Step 2: fuse two-qubit gates
  it = graph.tile_begin();
  do {
    auto nextIt = std::next(it);
    if (nextIt == graph.tile_end())
      break; // no next row to fuse with

    for (int q = 0; q < graph.nQubits(); ++q) {
      auto* gateL = (*it)[q];
      if (gateL == nullptr || gateL->nQubits() != 2)
        continue;
      auto* gateR = (*nextIt)[q];
      if (gateR == nullptr || gateR->nQubits() != 2)
        continue;

      // We have two two-qubit gates now
      const auto q0L = gateL->qubits()[0];
      const auto q1L = gateL->qubits()[1];
      const auto q0R = gateR->qubits()[0];
      const auto q1R = gateR->qubits()[1];
      // They act on the same qubits. Always fuse them.
      if (q0L == q0R && q1L == q1R) {
        graph.fuseAndInsertDiffRow(it, q);
        nFused++;
        continue;
      }
      // They act on different qubits. Directly fusing them gives a 3-qubit
      // gate. We check commutation of gateR with another gate and see if we
      // can swap them.
      if (swaTol <= 0.0)
        continue; // swapping analysis is disabled
      QuantumGate* gateToSwap = nullptr;
      auto nextNextIt = std::next(nextIt);
      if (nextNextIt != graph.tile_end()) {
        auto* nextGate = (*nextNextIt)[q];
        if (nextGate != nullptr && nextGate->nQubits() == 2 &&
            nextGate->qubits()[0] == q0L &&
            nextGate->qubits()[1] == q1L && 
            cast::isCommuting(gateR, nextGate)) {
          gateToSwap = nextGate;
        }
      }
      if (gateToSwap == nullptr) {
        LLVM_DEBUG(
          std::cerr << "Cannot swap gates at (" 
                    << graph.gateId(gateL) << ", " 
                    << graph.gateId(gateR) << ") with next gate\n";
        );
        continue; // cannot swap, skip
      }

      // We can swap gateR and gateToSwap
      graph.swapGates(nextIt, q);
      graph.squeeze(nextIt);
      // After swapping, it may not be true that nextIt == std::next(it).
      // But we are sure gateL still equals to (*it)[q].
      assert((*it)[q] == gateL && "GateL should not be changed after swapping");
      assert((*(std::next(it)))[q] == gateToSwap &&
             "GateToSwap should be in the next row after swapping");
      LLVM_DEBUG(
        std::cerr << "Swapped gates at (" 
                  << graph.gateId(gateL) << ", " 
                  << graph.gateId(gateR) << ") with gate "
                  << graph.gateId(gateToSwap) << "\n";
      );
      graph.fuseAndInsertDiffRow(it, q);
      nFused++;
    }
  } while (++it != graph.tile_end());

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
  return gates;
}

int applyFusionCFOPass_PriorIf(CircuitGraphNode* priorCGNode,
                               IfMeasureNode* ifNode, 
                               const FusionConfig& config,
                               int max_k_candidate) {
  assert(priorCGNode != nullptr);
  assert(ifNode != nullptr);

  if (priorCGNode->tile().empty())
    return false; // nothing to do

  auto thenCGNode = getOrAppendCGNodeToCompoundNodeBack(ifNode->thenBody);
  auto elseCGNode = getOrAppendCGNodeToCompoundNodeBack(ifNode->elseBody);
  
  auto freeGates = collectCommutingGatesWithMeasurement(
    *priorCGNode, ifNode->qubit, config.swaTol);
      
  int nFused = 0;
  int i;
  int size = freeGates.size();
  // we need to handle the empty case. Otherwise the loop that tries to put
  // remaining gates back to the prior CG node will not work.
  if (size == 0) {
    return nFused;
  }

  for (i = 0; i < size; ++i) {
    auto gate = freeGates[i];
    // insert the gate to both then and else CG nodes
    auto thenRowIt = thenCGNode->insertGate(gate, thenCGNode->tile_begin());
    auto elseRowIt = elseCGNode->insertGate(gate, elseCGNode->tile_begin());

    // start fusion on both CG nodes
    auto thenFused = cast::impl::startFusion(
        *thenCGNode, config, max_k_candidate, thenRowIt, gate->qubits()[0]);
    auto elseFused = cast::impl::startFusion(
        *elseCGNode, config, max_k_candidate, elseRowIt, gate->qubits()[0]);
    bool progress = (thenFused > 0 || elseFused > 0);
    if (thenFused > 0) {
      nFused += thenFused;
      thenCGNode->squeeze(); 
    }
    if (elseFused > 0) {;
      nFused += elseFused;
      elseCGNode->squeeze();
    }
    if (progress) {
      // we have successfully fused the gate, continue to the next gate
      continue;
    }
    // no effect
    // put the gate back to the prior CG node
    thenCGNode->removeGate(thenRowIt, gate->qubits()[0]);
    elseCGNode->removeGate(elseRowIt, gate->qubits()[0]);
    thenCGNode->squeeze();
    elseCGNode->squeeze();
    break;
  }

  // put the remaining gates back to the prior CG node
  for (int j = size - 1; j >= i; --j) {
    priorCGNode->insertGate(freeGates[j]);
  }
  priorCGNode->squeeze();

  return nFused;
}

int applyFusionCFOPass_IfJoin(IfMeasureNode* ifNode, 
                              CircuitGraphNode* joinCGNode,
                              const FusionConfig& config,
                              int max_k_candidate) {
  assert(joinCGNode != nullptr);
  assert(ifNode != nullptr);
  int nFused = 0;
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
        *thenCGNode, config, max_k_candidate, std::prev(thenRowIt), q);
    }
    if (elseRowIt != elseCGNode->tile_begin()) {
      elseFused = cast::impl::startFusion(
        *elseCGNode, config, max_k_candidate, std::prev(elseRowIt), q);
    }
    if (thenFused > 0 || elseFused > 0) {
      nFused += thenFused;
      nFused += elseFused;
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
  return nFused;
}

} // end of anonymous namespace

void cast::impl::applyGateFusion(ir::CircuitGraphNode& graph,
                                 const FusionConfig& config) {
  int max_k_candidate = (config.incrementScheme ? 2 : config.maxKOverride);
  while (true) {
    int nFusedThisRound = 0;
    auto it = graph.tile_begin();
    // we need to query graph.tile_end() every time, because impl::startFusion
    // may change graph tile
    while (it != graph.tile_end()) {
      for (int q = 0; q < graph.nQubits(); ++q) {
        nFusedThisRound += cast::impl::startFusion(
          graph, config, max_k_candidate, it, q);
      }
      ++it;
    }
    if (nFusedThisRound > 0)
      graph.squeeze();
      
    if (config.multiTraversal && nFusedThisRound > 0) {
      // continue without increasing max_k_candidate
      continue;
    }
    ++max_k_candidate;
    if (max_k_candidate > config.maxKOverride) {
      // we have reached the maximum k candidate
      break;
    }
  }
}

int cast::impl::applyCFOFusion(ir::CircuitNode& circuit,
                               const FusionConfig& config,
                               int max_k_candidate) {
  int nFused = 0;
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
        nFused += applyFusionCFOPass_PriorIf(
          priorCGNode, ifNode, config, max_k_candidate);
      }
    }
    else if (auto* ifNode = llvm::dyn_cast<IfMeasureNode>(prevIt->get())) {
      if (auto* joinCGNode = llvm::dyn_cast<CircuitGraphNode>(curIt->get())) {
        nFused += applyFusionCFOPass_IfJoin(
          ifNode, joinCGNode, config, max_k_candidate);
      }
    }
  } // end of while loop
  return nFused;
}

#undef DEBUG_TYPE