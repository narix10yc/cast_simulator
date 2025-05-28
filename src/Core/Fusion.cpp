#include "cast/Legacy/CircuitGraph.h"
#include "cast/Fusion.h"

#include "utils/iocolor.h"

#define DEBUG_TYPE "fusion"
#include "llvm/Support/Debug.h"
// #define LLVM_DEBUG(X) X

using namespace IOColor;
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

using tile_iter_t = legacy::CircuitGraph::tile_iter_t;

int getKAfterFusion(const legacy::GateBlock* blockA,
                    const legacy::GateBlock* blockB) {
  int count = 0;
  auto itA = blockA->quantumGate->qubits.begin();
  auto itB = blockB->quantumGate->qubits.begin();
  const auto endA = blockA->quantumGate->qubits.end();
  const auto endB = blockB->quantumGate->qubits.end();
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

/// Fuse two blocks on the same row. \c graph will be updated.
/// Notice this function will NOT delete old blocks.
/// @return fused block
legacy::GateBlock* fuseAndInsertSameRow(
    legacy::CircuitGraph& graph, tile_iter_t iter,
    legacy::GateBlock* blockA, legacy::GateBlock* blockB) {
  auto* blockFused = graph.acquireGateBlock(blockA, blockB);
  for (const auto& wireA : blockA->wires)
    (*iter)[wireA.qubit] = blockFused;
  for (const auto& wireB : blockB->wires)
    (*iter)[wireB.qubit] = blockFused;
  return blockFused;
}

/// Fuse two blocks on different rows. \c graph will be updated.
/// Notice this function will NOT delete old blocks.
/// @return the tile iterator of the fused block
tile_iter_t fuseAndInsertDiffRow(
    legacy::CircuitGraph& graph, tile_iter_t iterL, int qubit) {
  assert(iterL != nullptr);
  auto* blockL = (*iterL)[qubit];
  assert(blockL != nullptr);
  auto iterR = iterL.next();
  assert(iterR != nullptr);
  auto* blockR = (*iterR)[qubit];
  assert(blockR);

  auto* blockFused = graph.acquireGateBlock(blockL, blockR);
  for (const auto& wireL : blockL->wires)
    (*iterL)[wireL.qubit] = nullptr;
  for (const auto& wireR : blockR->wires)
    (*iterR)[wireR.qubit] = nullptr;

  // prioritize iterR > iterL > between iterL and iterR
  if (graph.isRowVacant(*iterR, blockFused)) {
    for (const auto& wireF : blockFused->wires)
      (*iterR)[wireF.qubit] = blockFused;
    return iterR;
  }
  if (graph.isRowVacant(*iterL, blockFused)) {
    for (const auto& wireF : blockFused->wires)
      (*iterL)[wireF.qubit] = blockFused;
    return iterL;
  }
  auto iterInserted = graph.tile().emplace_insert(iterL);
  for (const auto& wireF : blockFused->wires)
    (*iterInserted)[wireF.qubit] = blockFused;
  return iterInserted;
}

struct TentativeFusedItem {
  legacy::GateBlock* block;
  tile_iter_t iter;
};

/// @return Number of fused blocks
int startFusion(
    legacy::CircuitGraph& graph, const FusionConfig& config,
    const CostModel* costModel,
    const int maxK, tile_iter_t curIt, const int qubit) {
  auto* fusedBlock = (*curIt)[qubit];
  auto fusedIt = curIt;
  if (fusedBlock == nullptr)
    return 0;

  std::vector<TentativeFusedItem> fusedBlocks;
  fusedBlocks.reserve(8);
  fusedBlocks.emplace_back(fusedBlock, fusedIt);

  const auto checkFuseable = [&](legacy::GateBlock* candidateBlock) {
    if (candidateBlock == nullptr)
      return false;
    if (std::ranges::find_if(fusedBlocks,
        [b=candidateBlock](const auto& item) {
          return item.block == b;
        }) != fusedBlocks.end()) {
      return false;
    }
    return getKAfterFusion(fusedBlock, candidateBlock) <= maxK;
  };

  // Start with same-row blocks
  for (int q = qubit+1; q < graph.nQubits; ++q) {
    auto* candidateBlock = (*curIt)[q];
    if (candidateBlock == fusedBlock)
      continue;
    if (checkFuseable(candidateBlock) == false)
      continue;

    // candidateBlock is accepted
    fusedBlocks.emplace_back(candidateBlock, curIt);
    fusedBlock = fuseAndInsertSameRow(graph, curIt, fusedBlock, candidateBlock);
  }

  assert(curIt == fusedIt);

  // TODO: check logic here
  bool progress;
  do {
    curIt = fusedIt.next();
    if (curIt == graph.tile_end())
      break;

    progress = false;
    for (const auto& q : fusedBlock->quantumGate->qubits) {
      auto* candidateBlock = (*curIt)[q];
      if (checkFuseable(candidateBlock) == false)
        continue;
      // candidateBlock is accepted
      fusedBlocks.emplace_back(candidateBlock, curIt);
      fusedIt = fuseAndInsertDiffRow(graph, curIt.prev(), q);
      fusedBlock = (*fusedIt)[candidateBlock->quantumGate->qubits[0]];
      progress = true;
      break;
    }
  } while (progress == true);

  assert(fusedBlocks.size() > 0);
  if (fusedBlocks.size() == 1)
    return 0;

  assert(fusedIt != graph.tile_end());

  // Check benefit
  double oldTime = 0.0;
  for (const auto& [block, iter] : fusedBlocks) {
    oldTime += costModel->computeGiBTime(
      *block->quantumGate, config.precision, config.nThreads);
  }
  double newTime = costModel->computeGiBTime(
    *fusedBlock->quantumGate, config.precision, config.nThreads);
  double benefit = oldTime / newTime - 1.0;
  LLVM_DEBUG(
    utils::printArray(std::cerr,
      llvm::ArrayRef(fusedBlocks.data(), fusedBlocks.size()),
      [](std::ostream& os, const TentativeFusedItem& item) {
        os << item.block->id << "(" << item.block->nQubits()
           << "," << item.block->quantumGate->opCount(1e-8) << ")";
      });
    std::cerr << " => " << fusedBlock->id << "(" << fusedBlock->nQubits()
              << "," << fusedBlock->quantumGate->opCount(1e-8) << "). "
              << "Benefit = " << benefit << "; ";
  );

  if (benefit < config.benefitMargin) {
    // undo this fusion
    LLVM_DEBUG(std::cerr << "Rejected\n");
    // LLVM_DEBUG(
    //   graph.print(std::cerr << "-- Before Rejection --\n", 2) << "\n";
    // );
    for (const auto& q : fusedBlock->quantumGate->qubits) {
      assert((*fusedIt)[q] == fusedBlock);
      (*fusedIt)[q] = nullptr;
    }
    for (const auto& [block, iter] : fusedBlocks) {
      for (const auto& q : block->quantumGate->qubits)
        (*iter)[q] = block;
    }
    // LLVM_DEBUG(
    //   graph.print(std::cerr << "-- After Rejection --\n", 2) << "\n";
    // );

    graph.releaseGateBlock(fusedBlock);
    return 0;
  }
  // accept this fusion
  for (const auto& [block, iter] : fusedBlocks)
    graph.releaseGateBlock(block);
  LLVM_DEBUG(std::cerr << "Accepted\n";);
  return fusedBlocks.size() - 1;
}

} // anonymous namespace

void cast::applyGateFusion(
    const FusionConfig& config, const CostModel* costModel,
    legacy::CircuitGraph& graph, int max_k) {
  int curMaxK = 2;
  int nFused = 0;
  do {
    nFused = 0;
    auto it = graph.tile_begin();
    int q = 0;
    while (it != graph.tile_end()) {
      for (q = 0; q < graph.nQubits; ++q) {
        nFused += startFusion(
          graph, config, costModel, curMaxK, it, q);;
      }
      ++it;
    }
    graph.squeeze();
  } while (nFused > 0 && ++curMaxK <= max_k);
}

