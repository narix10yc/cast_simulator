#include "cast/FPGA/FPGAFusion.h"
#include "utils/utils.h"
#include "utils/iocolor.h"

using namespace cast;
using namespace cast::fpga;

namespace {


// Compute a candidate gate for fusion.
// @return nullptr if fusion is not possible.
QuantumGatePtr computeCandidate(
    const FPGAFusionConfig& config,
    const QuantumGate* lhs, const QuantumGate* rhs) {
  if (lhs == nullptr || rhs == nullptr)
    return nullptr;
  if (lhs == rhs)
    return nullptr;

  // set up qubits of candidate block
  auto rstQubits = lhs->qubits();
  for (const auto& q : rhs->qubits())
    utils::push_back_if_not_present(rstQubits, q);

  auto lhsCate = getFPGAGateCategory(lhs, config.tolerances);
  auto rhsCate = getFPGAGateCategory(rhs, config.tolerances);

  // check fusion condition
  // 1. ignore non-comp gates
  if (config.ignoreSingleQubitNonCompGates) {
    if (lhsCate.is(FPGAGateCategory::fpgaNonComp)) {
      // std::cerr << CYAN_FG << "Omitted because LHS block "
      //   << lhs->id << " is a non-comp gate\n" << RESET;
      // lhs->quantumGate->gateMatrix.printCMat(std::cerr) << "\n";
      return nullptr;
    }
    if (rhsCate.is(FPGAGateCategory::fpgaNonComp)) {
      // std::cerr << CYAN_FG << "Omitted because RHS block "
      //   << rhs->id << " is a non-comp gate\n" << RESET;
      // rhs->quantumGate->gateMatrix.printCMat(std::cerr) << "\n";
      return nullptr;
    }
  }

  // 2. multi-qubit gates: only fuse when unitary perm
  // We do not have kernels for multi-qubit non-unitary-perm gates.
  if (lhsCate.isNot(FPGAGateCategory::fpgaSingleQubit)) {
    assert(lhsCate.is(FPGAGateCategory::fpgaUnitaryPerm) &&
           "LHS gate is multi-qubit non-unitary-perm.");
  }
  if (rhsCate.isNot(FPGAGateCategory::fpgaSingleQubit)) {
    assert(rhsCate.is(FPGAGateCategory::fpgaUnitaryPerm) &&
           "RHS gate is multi-qubit non-unitary-perm.");
  }

  // 3. check resulting size
  // 3.1 resulting gate size is larger than max size, reject
  if (rstQubits.size() > config.maxUnitaryPermutationSize) {
    // std::cerr << YELLOW_FG << "Rejected because the candidate block size is
    // too large\n" << RESET;
    return nullptr;
  }
  // 3.2 resulting gate size is okay, accept if it is unitary perm
  if (rstQubits.size() > 1) {
    if (lhsCate.isNot(FPGAGateCategory::fpgaUnitaryPerm) ||
        rhsCate.isNot(FPGAGateCategory::fpgaUnitaryPerm)) {
      // std::cerr << YELLOW_FG
      //   << "Rejected because the resulting gate "
      //  "is multi-qubit but not unitary perm\n" << RESET;
      return nullptr;
    }
  }

  // accept candidate
  // std::cerr << GREEN_FG << "Fusion accepted! " << "\n" << RESET;
  auto rstGate = cast::matmul(rhs, lhs);
  return rstGate;
}

using row_iterator = ir::CircuitGraphNode::row_iterator;

ConstQuantumGatePtr trySameWireFuse(
    ir::CircuitGraphNode& graph, row_iterator itLHS,
    int q, const FPGAFusionConfig& config) {
  assert(itLHS != graph.tile_end());
  const auto itRHS = std::next(itLHS);
  if (itRHS == graph.tile_end())
    return nullptr;

  auto lhs = (*itLHS)[q];
  auto rhs = (*itRHS)[q];

  if (!lhs || !rhs)
    return nullptr;

  // candidate gate
  auto cddGate = computeCandidate(config, lhs, rhs);
  if (cddGate == nullptr)
    return nullptr;

  graph.removeGate(itLHS, q);
  graph.removeGate(itRHS, q);
  graph.insertGate(cddGate, itLHS);
  return cddGate;
}

GateBlock* tryCrossWireFuse(cast::legacy::CircuitGraph& graph,
                            const tile_iter_t& tileIt,
                            int q, const FPGAFusionConfig& config) {
  auto block0 = (*tileIt)[q];
  if (block0 == nullptr)
    return nullptr;

  for (unsigned q1 = 0; q1 < graph.nQubits; q1++) {
    auto* block1 = (*tileIt)[q1];
    auto* fusedBlock = computeCandidate(config, block0, block1);
    if (fusedBlock == nullptr)
      continue;
    for (const auto q : fusedBlock->quantumGate->qubits) {
      (*tileIt)[q] = fusedBlock;
    }
    delete (block0);
    delete (block1);
    return fusedBlock;
  }
  return nullptr;
}
} // anonymous namespace

void cast::applyFPGAGateFusion(
    legacy::CircuitGraph& graph, const FPGAFusionConfig& config) {
  auto& tile = graph.tile();
  if (tile.size() < 2)
    return;

  GateBlock* lhsBlock;
  GateBlock* rhsBlock;

  bool hasChange = true;
  tile_iter_t tileIt;
  unsigned q = 0;
  // multi-traversal
  while (hasChange) {
    tileIt = tile.begin();
    hasChange = false;
    while (tileIt.next() != tile.end()) {
      // same wire (connected consecutive) fuse
      q = 0;
      while (q < graph.nQubits) {
        if ((*tileIt)[q] == nullptr) {
          q++;
          continue;
        }
        if ((*tileIt.next())[q] == nullptr) {
          graph.repositionBlockDownward(tileIt, q++);
          continue;
        }
        auto* fusedBlock = trySameWireFuse(graph, tileIt, q, config);
        if (fusedBlock == nullptr)
          q++;
        else
          hasChange = true;
      }
      // cross wire (same row) fuse
      q = 0;
      while (q < graph.nQubits) {
        auto* fusedBlock = tryCrossWireFuse(graph, tileIt, q, config);
        if (fusedBlock == nullptr)
          q++;
        else
          hasChange = true;
      }
      tileIt++;
    }
    graph.eraseEmptyRows();
    // graph.updateTileUpward();
    if (!config.multiTraverse)
      break;
  }
}
