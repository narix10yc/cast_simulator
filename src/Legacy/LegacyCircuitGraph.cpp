#include "cast/Legacy/CircuitGraph.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

#include <chrono>
#include <deque>
#include <iomanip>
#include <map>
#include <numeric>
#include <thread>

using namespace IOColor;
using namespace cast::legacy;

// static member
int CircuitGraphContext::GateNodeCount = 0;
int CircuitGraphContext::GateBlockCount = 0;

GateNode::GateNode(std::shared_ptr<QuantumGate> gate, const CircuitGraph& graph)
    : id(CircuitGraphContext::GateNodeCount++), quantumGate(gate) {
  connections.reserve(quantumGate->nQubits());
  for (const auto q : quantumGate->qubits) {
    connections.emplace_back(q, nullptr, nullptr);
    auto it = graph.tile().tail_iter();
    while (it != nullptr && (*it)[q] == nullptr)
      --it;
    if (it != nullptr) {
      auto* lhsBlockWire = (*it)[q]->findWire(q);
      assert(lhsBlockWire != nullptr);
      lhsBlockWire->rhsEntry->connect(this, q);
    }
  }
}

GateBlock::GateBlock()
    : id(CircuitGraphContext::GateBlockCount++), quantumGate(nullptr) {}

GateBlock::GateBlock(GateNode* gateNode)
    : id(CircuitGraphContext::GateBlockCount++),
      quantumGate(gateNode->quantumGate) {
  wires.reserve(quantumGate->nQubits());
  for (const auto& data : gateNode->connections)
    wires.emplace_back(data.qubit, gateNode, gateNode);
}

void GateNode::connect(GateNode* rhsGate, int q) {
  assert(rhsGate);
  auto* myIt = findConnection(q);
  assert(myIt);
  auto* rhsIt = rhsGate->findConnection(q);
  assert(rhsIt);

  myIt->rhsGate = rhsGate;
  rhsIt->lhsGate = this;
}

void CircuitGraph::clear() {
  auto allBlocks = getAllBlocks();
  for (auto* block : allBlocks)
    releaseGateBlock(block);
  _tile.clear();
}

void CircuitGraph::QFTCircuit(int nQubits, CircuitGraph& graph) {
  for (int q = 0; q < nQubits; ++q) {
    graph.appendGate(std::make_shared<QuantumGate>(QuantumGate::H(q)));
    for (int l = q + 1; l < nQubits; ++l) {
      double angle = M_PI_2 * std::pow(2.0, q - l);
      graph.appendGate(std::make_shared<QuantumGate>(
          QuantumGate(GateMatrix::FromName("cp", {angle}), {q, l})));
    }
  }
}

// CircuitGraph CircuitGraph::ALACircuit(int nQubits, int nrounds) {
//   assert(0 && "Not Implemented");
//   CircuitGraph graph;
//   return graph;
// }

// CircuitGraph CircuitGraph::GetTestCircuit(
//     const GateMatrix& gateMatrix, int nQubits, int nrounds) {
//   CircuitGraph graph;
//   auto nQubitsGate = gateMatrix.nQubits();
//
//   for (int r = 0; r < nrounds; r++) {
//     for (int q = 0; q < nQubits; q++) {
//       graph.addGate(std::make_unique<QuantumGate>(
//         gateMatrix,
//         std::initializer_list<int>{q, (q + 1) % nQubits, (q + 2) % nQubits});
//     }
//   }
//   return graph;
// }

GateBlock* CircuitGraph::acquireGateBlock(GateBlock* lhsBlock,
                                          GateBlock* rhsBlock) {
  auto quantumGate = std::make_shared<QuantumGate>(
      lhsBlock->quantumGate->lmatmul(*rhsBlock->quantumGate));

  auto* gateBlock = _context->gateBlockPool.acquire();
  gateBlock->quantumGate = quantumGate;

  // setup wires
  for (const auto& q : quantumGate->qubits) {
    const auto* lWire = lhsBlock->findWire(q);
    const auto* rWire = rhsBlock->findWire(q);
    if (lWire && rWire)
      gateBlock->wires.emplace_back(q, lWire->lhsEntry, rWire->rhsEntry);
    else if (lWire)
      gateBlock->wires.emplace_back(q, lWire->lhsEntry, lWire->rhsEntry);
    else {
      assert(rWire);
      gateBlock->wires.emplace_back(q, rWire->lhsEntry, rWire->rhsEntry);
    }
  }

  return gateBlock;
}

CircuitGraph::tile_iter_t CircuitGraph::insertBlock(tile_iter_t it,
                                                    GateBlock* block) {

  // print(std::cerr << "About to insertBlock " << block->id << "\n", 2) <<
  // "\n";
  assert(it != nullptr);
  assert(block != nullptr);

  const auto& qubits = block->quantumGate->qubits;
  assert(!qubits.empty());

  // try insert to current row
  if (isRowVacant(*it, block)) {
    for (const auto& q : qubits)
      (*it)[q] = block;
    return it;
  }

  // try insert to next row
  ++it;
  if (it == nullptr) {
    _tile.emplace_back();
    auto itTail = _tile.tail_iter();
    for (const auto& q : qubits)
      (*itTail)[q] = block;
    return itTail;
  }

  if (isRowVacant(*it, block)) {
    for (const auto& q : qubits)
      (*it)[q] = block;
    return it;
  }

  // insert between current and next row
  it = _tile.emplace_insert(it);
  for (const auto& q : qubits)
    (*it)[q] = block;
  return it;
}

void CircuitGraph::appendGate(std::shared_ptr<QuantumGate> quantumGate) {
  assert(quantumGate != nullptr);
  // update nQubits
  for (const auto& q : quantumGate->qubits) {
    if (q >= nQubits)
      nQubits = q + 1;
  }

  // create gate and setup connections
  auto* gateNode = acquireGateNodeForward(quantumGate, *this);

  // create block and insert to the tile
  // TODO: this is slightly inefficient as the block may be assigned twice
  auto* block = acquireGateBlockForward(gateNode);
  auto it = insertBlock(_tile.tail_iter(), block);
  repositionBlockUpward(it, block->quantumGate->qubits[0]);
}

std::vector<GateBlock*> CircuitGraph::getAllBlocks() const {
  std::vector<GateBlock*> allBlocks;
  std::vector<GateBlock*> rowBlocks;
  for (const auto& row : _tile) {
    rowBlocks.clear();
    for (const auto& block : row) {
      if (block == nullptr)
        continue;
      if (std::ranges::find(rowBlocks, block) == rowBlocks.end())
        rowBlocks.push_back(block);
    }
    for (const auto& block : rowBlocks)
      allBlocks.push_back(block);
  }
  return allBlocks;
}

CircuitGraph::list_node_t* CircuitGraph::repositionBlockUpward(list_node_t* ln,
                                                               int q) {
  assert(ln != nullptr);
  auto* block = ln->data[q];
  assert(block && "Cannot reposition a null block");
  // find which row fits the block
  auto* newln = ln;
  const auto* const head = _tile.head();
  if (newln == head)
    return newln;

  do {
    if (isRowVacant(newln->prev->data, block))
      newln = newln->prev;
    else
      break;
  } while (newln != head);

  // put block into the new position
  if (newln == ln)
    return ln;
  for (const auto& data : block->wires) {
    const auto& i = data.qubit;
    ln->data[i] = nullptr;
    newln->data[i] = block;
  }

  return newln;
}

CircuitGraph::list_node_t*
CircuitGraph::repositionBlockDownward(list_node_t* ln, int q) {
  assert(ln != nullptr);
  auto* block = ln->data[q];
  assert(block && "Cannot reposition a null block");
  // find which row fits the block
  auto* newln = ln;
  const auto* const tail = _tile.tail();
  if (newln == tail)
    return newln;

  do {
    if (isRowVacant(newln->next->data, block))
      newln = newln->next;
    else
      break;
  } while (newln != tail);

  // put block into the new position
  if (newln == ln)
    return ln;
  for (const auto& data : block->wires) {
    const auto& i = data.qubit;
    ln->data[i] = nullptr;
    newln->data[i] = block;
  }

  return newln;
}

void CircuitGraph::eraseEmptyRows() {
  auto it = _tile.cbegin();
  while (it != nullptr) {
    bool empty = true;
    for (unsigned q = 0; q < nQubits; q++) {
      if ((*it)[q] != nullptr) {
        empty = false;
        break;
      }
    }
    if (empty)
      it = _tile.erase(it);
    else
      ++it;
  }
}

void CircuitGraph::squeeze() {
  eraseEmptyRows();
  auto it = _tile.begin();
  while (it != nullptr) {
    for (unsigned q = 0; q < nQubits; q++) {
      if ((*it)[q])
        repositionBlockUpward(it, q);
    }
    ++it;
  }
  eraseEmptyRows();
}

std::ostream& CircuitGraph::print(std::ostream& os, int verbose) const {
  if (_tile.empty())
    return os << "<empty tile>\n";
  int width = static_cast<int>(std::log10(_context->GateBlockCount) + 1);
  if ((width & 1) == 0)
    width++;

  const std::string vbar =
      std::string(width / 2, ' ') + "|" + std::string(width / 2 + 1, ' ');

  for (const auto& row : _tile) {
    if (verbose > 1)
      os << &row << ": ";
    for (unsigned q = 0; q < nQubits; q++) {
      if (const auto* block = row[q]; block != nullptr)
        os << std::setw(width) << std::setfill('0') << block->id << " ";
      else
        os << vbar;
    }
    os << "\n";
  }
  return os;
}

//
// std::ostream& GateBlock::displayInfo(std::ostream& os) const {
//   os << "Block " << id << ": [";
//   for (const auto& data : wires) {
//     os << "(" << data.qubit << ":";
//     GateNode* gate = data.lhsEntry;
//     assert(gate);
//     os << gate->id << ",";
//     while (gate != data.rhsEntry) {
//       gate = gate->findRHS(data.qubit);
//       assert(gate);
//       os << gate->id << ",";
//     }
//     os << "),";
//   }
//   return os << "]\n";
// }
//
// std::vector<int> CircuitGraph::getBlockSizes() const {
//   std::vector<int> sizes(nQubits + 1, 0);
//   const auto allBlocks = getAllBlocks();
//   int largestSize = 0;
//   for (const auto* b : allBlocks) {
//     auto blocknQubits = b->nQubits();
//     sizes[blocknQubits]++;
//     if (blocknQubits > largestSize)
//       largestSize = blocknQubits;
//   }
//   sizes.resize(largestSize + 1);
//   return sizes;
// }
//
// std::vector<std::vector<int>> CircuitGraph::getBlockOpCountHistogram() const
// {
//   const auto allBlocks = getAllBlocks();
//   int largestSize = 0;
//   for (const auto* b : allBlocks) {
//     auto blocknQubits = b->nQubits();
//     if (blocknQubits > largestSize)
//       largestSize = blocknQubits;
//   }
//   std::vector<std::vector<int>> hist(largestSize + 1);
//   for (unsigned q = 1; q < largestSize + 1; q++)
//     hist[q].resize(q, 0);
//
//   for (const auto* b : allBlocks) {
//     const int q = b->nQubits();
//     int catagory = 0;
//     int opCount = b->quantumGate->opCount();
//     while ((1 << (2 * catagory + 3)) < opCount)
//       catagory++;
//
//     hist[q][catagory]++;
//   }
//   return hist;
// }
//
// std::ostream& CircuitGraph::displayInfo(std::ostream& os, int verbose) const
// {
//   os << CYAN_FG << "=== CircuitGraph Info (verbose " << verbose << ") ===\n"
//      << RESET;
//
//   os << "- Number of Gates:  " << countGates() << "\n";
//   const auto allBlocks = getAllBlocks();
//   auto nBlocks = allBlocks.size();
//   os << "- Number of Blocks: " << nBlocks << "\n";
//   auto totalOp = countTotalOps();
//   os << "- Total Op Count:   " << totalOp << "\n";
//   os << "- Average Op Count: " << std::fixed << std::setprecision(1)
//      << static_cast<double>(totalOp) / nBlocks << "\n";
//   os << "- Circuit Depth:    " << _tile.size() << "\n";
//
//   if (verbose > 3) {
//     os << "- Block Sizes Count:\n";
//     std::vector<std::vector<int>> vec(nQubits + 1);
//     const auto allBlocks = getAllBlocks();
//     for (const auto* block : allBlocks)
//       vec[block->nQubits()].push_back(block->id);
//
//     for (unsigned q = 1; q < vec.size(); q++) {
//       if (vec[q].empty())
//         continue;
//       os << "  " << q << "-qubit: count " << vec[q].size() << " ";
//       utils::printVector(vec[q], os) << "\n";
//     }
//   } else if (verbose > 2) {
//     os << "- Block Statistics:\n";
//     const auto hist = getBlockOpCountHistogram();
//     for (unsigned q = 1; q < hist.size(); q++) {
//       auto count = std::reduce(hist[q].begin(), hist[q].end());
//       if (count == 0)
//         continue;
//       os << "  " << q << "-qubit count = " << count << "; hist: ";
//       utils::printVector(hist[q], os) << "\n";
//     }
//   } else if (verbose > 1) {
//     os << "- Block Sizes Count:\n";
//     const auto sizes = getBlockSizes();
//     for (unsigned q = 1; q < sizes.size(); q++) {
//       if (sizes[q] <= 0)
//         continue;
//       os << "  " << q << "-qubit: " << sizes[q] << "\n";
//     }
//   }
//
//   os << CYAN_FG << "=====================================\n" << RESET;
//   return os;
// }

std::vector<GateNode*> GateBlock::getOrderedGates() const {
  std::deque<GateNode*> queue;
  // vector should be more efficient as we expect small sizes here
  std::vector<GateNode*> gates;
  gates.reserve(8);
  for (const auto& data : wires) {
    if (std::ranges::find(queue, data.rhsEntry) == queue.end())
      queue.push_back(data.rhsEntry);
  }

  while (!queue.empty()) {
    const auto& gate = queue.back();
    if (std::ranges::find(gates, gate) != gates.end()) {
      queue.pop_back();
      continue;
    }
    std::vector<GateNode*> higherPriorityGates;
    for (const auto& data : this->wires) {
      if (gate == data.lhsEntry)
        continue;
      const auto* connection = gate->findConnection(data.qubit);
      if (connection == nullptr)
        continue;
      assert(connection->lhsGate);
      if (std::ranges::find(gates, connection->lhsGate) == gates.end())
        higherPriorityGates.push_back(connection->lhsGate);
    }

    if (higherPriorityGates.empty()) {
      queue.pop_back();
      gates.push_back(gate);
    } else {
      for (const auto& g : higherPriorityGates)
        queue.push_back(g);
    }
  }
  return gates;
}

bool GateBlock::isSingleton() const {
  const auto* node = wires[0].lhsEntry;
  for (const auto& wire : wires) {
    if (wire.lhsEntry != wire.rhsEntry)
      return false;
    if (wire.lhsEntry != node)
      return false;
  }
  return true;
}

// void CircuitGraph::relabelBlocks() const {
//   int count = 0;
//   auto allBlocks = getAllBlocks();
//   for (auto* block : allBlocks)
//     block->id = (count++);
// }

void CircuitGraph::deepCopy(CircuitGraph& other) const {
  assert(false && "Not Implemented");
}
