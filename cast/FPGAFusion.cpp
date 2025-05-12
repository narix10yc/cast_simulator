#include "cast/CircuitGraph.h"
#include "cast/FPGAInst.h"
#include "cast/Fusion.h"
#include "cast/LegacyQuantumGate.h"
#include "utils/iocolor.h"

using namespace cast;
using namespace cast::fpga;
using namespace IOColor;

using tile_iter_t = CircuitGraph::tile_iter_t;

namespace {

GateBlock* computeCandidate(
    const FPGAFusionConfig& config,
    const GateBlock* lhs, const GateBlock* rhs) {
  if (lhs == nullptr || rhs == nullptr)
    return nullptr;
  if (lhs == rhs)
    return nullptr;

  assert(lhs->quantumGate != nullptr);
  assert(rhs->quantumGate != nullptr);

  // candidate block
  auto block = new GateBlock();

  // std::cerr << "Trying to fuse "
  //   << "lhs " << lhs->id << " and rhs " << rhs->id
  //   << " => candidate block " << block->id << "\n";

  // set up qubits of candidate block
  std::vector<int> blockQubits;
  for (const auto& lData : lhs->wires) {
    const auto& q = lData.qubit;

    GateNode* lhsEntry = lData.lhsEntry;
    GateNode* rhsEntry;
    auto it = rhs->findWire(q);
    if (it == nullptr)
      rhsEntry = lData.rhsEntry;
    else
      rhsEntry = it->rhsEntry;

    assert(lhsEntry);
    assert(rhsEntry);

    block->wires.push_back({q, lhsEntry, rhsEntry});
    blockQubits.push_back(q);
  }
  for (const auto& rData : rhs->wires) {
    const auto& q = rData.qubit;
    if (lhs->findWire(q) == nullptr) {
      block->wires.push_back(rData);
      blockQubits.push_back(q);
    }
  }

  auto lhsCate = getFPGAGateCategory(*lhs->quantumGate, config.tolerances);
  auto rhsCate = getFPGAGateCategory(*rhs->quantumGate, config.tolerances);

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
  if (lhsCate.isNot(FPGAGateCategory::fpgaSingleQubit)) {
    assert(lhsCate.is(FPGAGateCategory::fpgaUnitaryPerm) &&
           "We do not have kernel for multi-qubit non-unitary-perm gates");
  }
  if (rhsCate.isNot(FPGAGateCategory::fpgaSingleQubit)) {
    assert(rhsCate.is(FPGAGateCategory::fpgaUnitaryPerm) &&
           "We do not have kernel for multi-qubit non-unitary-perm gates");
  }

  if (blockQubits.size() > config.maxUnitaryPermutationSize) {
    // std::cerr << YELLOW_FG << "Rejected because the candidate block size is
    // too large\n" << RESET;
    return nullptr;
  }

  if (blockQubits.size() > 1) {
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

  block->quantumGate = std::make_unique<LegacyQuantumGate>(
      rhs->quantumGate->lmatmul(*lhs->quantumGate));

  return block;
}

GateBlock* trySameWireFuse(
    CircuitGraph& graph, tile_iter_t itLHS,
    int qubit, const FPGAFusionConfig& config) {
  assert(itLHS != graph.tile_end());
  const auto itRHS = itLHS.next();
  if (itRHS == graph.tile_end())
    return nullptr;

  GateBlock* lhs = (*itLHS)[qubit];
  GateBlock* rhs = (*itRHS)[qubit];

  if (!lhs || !rhs)
    return nullptr;

  GateBlock* block = computeCandidate(config, lhs, rhs);
  if (block == nullptr)
    return nullptr;

  // std::cerr << BLUE_FG;
  // lhs->displayInfo(std::cerr);
  // rhs->displayInfo(std::cerr);
  // block->displayInfo(std::cerr) << RESET;

  for (const auto& q : lhs->quantumGate->qubits)
    (*itLHS)[q] = nullptr;
  for (const auto& q : rhs->quantumGate->qubits)
    (*itRHS)[q] = nullptr;

  delete (lhs);
  delete (rhs);

  // insert block to tile
  graph.insertBlock(itLHS, block);
  return block;
}

GateBlock* tryCrossWireFuse(CircuitGraph& graph, const tile_iter_t& tileIt,
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
    CircuitGraph& graph, const FPGAFusionConfig& config) {
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
