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

  auto lhsCate = getFPGAGateCategory(lhs, config.tolerance);
  auto rhsCate = getFPGAGateCategory(rhs, config.tolerance);

  // check fusion condition
  // 1. ignore non-comp gates
  if (config.ignoreSingleQubitNonCompGates) {
    if (lhsCate.is(FPGAGateCategory::NonComp)) {
      // std::cerr << CYAN_FG << "Omitted because LHS block "
      //   << lhs->id << " is a non-comp gate\n" << RESET;
      // lhs->quantumGate->gateMatrix.printCMat(std::cerr) << "\n";
      return nullptr;
    }
    if (rhsCate.is(FPGAGateCategory::NonComp)) {
      // std::cerr << CYAN_FG << "Omitted because RHS block "
      //   << rhs->id << " is a non-comp gate\n" << RESET;
      // rhs->quantumGate->gateMatrix.printCMat(std::cerr) << "\n";
      return nullptr;
    }
  }

  // 2. multi-qubit gates: only fuse when unitary perm
  // We do not have kernels for multi-qubit non-unitary-perm gates.
  if (lhsCate.isNot(FPGAGateCategory::SingleQubit)) {
    assert(lhsCate.is(FPGAGateCategory::UnitaryPerm) &&
           "LHS gate is multi-qubit non-unitary-perm.");
  }
  if (rhsCate.isNot(FPGAGateCategory::SingleQubit)) {
    assert(rhsCate.is(FPGAGateCategory::UnitaryPerm) &&
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
    if (lhsCate.isNot(FPGAGateCategory::UnitaryPerm) ||
        rhsCate.isNot(FPGAGateCategory::UnitaryPerm)) {
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

QuantumGatePtr tryCrossWireFuse(
    ir::CircuitGraphNode& graph,
    row_iterator rowItL, int q,
    const FPGAFusionConfig& config) {
  assert(rowItL != graph.tile_end());
  const auto rowItR = std::next(rowItL);
  if (rowItR == graph.tile_end())
    return nullptr;

  auto lhs = (*rowItL)[q];
  auto rhs = (*rowItR)[q];

  if (!lhs || !rhs)
    return nullptr;

  // candidate gate
  auto cddGate = computeCandidate(config, lhs, rhs);
  if (cddGate == nullptr)
    return nullptr;

  // accept candidate gate
  graph.replaceGatesOnConsecutiveRowsWith(cddGate, rowItL, q);
  return cddGate;
}

QuantumGatePtr trySameWireFuse(
    ir::CircuitGraphNode& graph,
    row_iterator rowIt, int q,
    const FPGAFusionConfig& config) {
  auto* gate0 = (*rowIt)[q];
  if (gate0 == nullptr)
    return nullptr;

  for (unsigned q1 = 0; q1 < graph.nQubits(); q1++) {
    auto* gate1 = (*rowIt)[q1];
    auto cddGate = computeCandidate(config, gate0, gate1);
    if (cddGate == nullptr)
      continue;
    // accept candidate gate
    graph.removeGate(rowIt, q);
    graph.removeGate(rowIt, q1);
    graph.insertGate(cddGate, rowIt);
    return cddGate;
  }
  return nullptr;
}
} // anonymous namespace

void cast::fpga::applyFPGAGateFusion(
    ir::CircuitGraphNode& graph, const FPGAFusionConfig& config) {
  // if (graph.tile().size() < 2)
    // return;

  const auto nQubits = graph.nQubits();
  bool hasChange = true;
  row_iterator rowIt;
  unsigned q = 0;
  // multi-traversal
  while (hasChange) {
    rowIt = graph.tile_begin();
    hasChange = false;
    while (std::next(rowIt) != graph.tile_end()) {
      // same wire (connected consecutive) fuse
      q = 0;
      while (q < nQubits) {
        if ((*rowIt)[q] == nullptr) {
          q++;
          continue;
        }
        auto fusedBlock = trySameWireFuse(graph, rowIt, q, config);
        if (fusedBlock == nullptr)
          q++;
        else
          hasChange = true;
      }
      // cross wire (same row) fuse
      q = 0;
      while (q < nQubits) {
        auto fusedBlock = tryCrossWireFuse(graph, rowIt, q, config);
        if (fusedBlock == nullptr)
          q++;
        else
          hasChange = true;
      }
      rowIt++;
    }
    graph.squeeze();
    if (!config.multiTraverse)
      break;
  }
}
