#include "cast/IR/IRNode.h"
#include "utils/iocolor.h"

#include <set>
#include <iomanip>

using namespace cast;
using namespace cast::ir;

int CircuitGraphNode::_gateMapId = 0;

// checks if all rows in the circuit graph have the same size
static bool checkConsistency_rowSize(const CircuitGraphNode& graph) {
  auto it = graph.tile_begin();
  const auto end = graph.tile_end();
  if (it == end)
    return true; // empty graph is consistent
  int row = -1;
  for (; it != end; ++it) {
    ++row;
    if (it->size() < graph.nQubits()) {
      std::cerr << BOLDRED("Err: ") << "In checkConsistency_rowSize(): "
        "The size of row " << row << " is too small: "
        "size = " << it->size()
        << " while graph nQubits is " << graph.nQubits() << "\n";
      return false;
    }
    for (int q = graph.nQubits(); q < it->size(); ++q) {
      if ((*it)[q] != nullptr) {
        std::cerr << BOLDRED("Err: ") << "In checkConsistency_rowSize(): "
          "Row " << row << " has a gate at qubit " << q
          << ", but nQubits is " << graph.nQubits() << "\n";
        return false;
      }
    }
  }
  return true;
}

// checks if gateMap (1). does not contain null gates, (2). does not have
// repetitions, (3). does not have negative ids
static bool checkConsistency_gateMap(const CircuitGraphNode& graph) {
  if (graph.gateMap().empty())
    return true;

  std::set<cast::QuantumGate*> gatePtrs;
  std::set<int> gateIds;
  for (const auto& [gatePtr, id] : graph.gateMap()) {
    if (gatePtr == nullptr) {
      std::cerr << BOLDRED("Err: ") << "In checkConsistency_gateMap(): "
        "Gate in gateMap is null.\n";
      return false;
    }
    if (id < 0) {
      std::cerr << BOLDRED("Err: ") << "In checkConsistency_gateMap(): "
        "Negative gate id found: " << id << "\n";
      return false;
    }
    if (!gatePtrs.insert(gatePtr.get()).second) {
      std::cerr << BOLDRED("Err: ") << "In checkConsistency_gateMap(): "
        "Duplicate gate pointer found in gateMap: "
        << (void*)(gatePtr.get()) << "\n";
      return false;
    }
    if (!gateIds.insert(id).second) {
      std::cerr << BOLDRED("Err: ") << "In checkConsistency_gateMap(): "
        "Duplicate gate id found in gateMap: " << id << "\n";
      return false;
    }
  }
  return true;
}

// checks if gates present in the tile are consistent with gateMap
static bool checkConsistency_gateMemory(const CircuitGraphNode& graph) {
  std::set<int> gatesInTile;
  int rowNumber = -1;
  const auto gateMapEnd = graph.gateMap().end();
  // Check if all gates in the tile are present in gateMap
  for (const auto& row : graph.tile()) {
    ++rowNumber;
    // more efficient to collect gates in a row first
    std::set<cast::QuantumGate*> gatesInRow;
    for (const auto& gate : row) {
      if (gate != nullptr)
        gatesInRow.insert(gate);
    }
    for (const auto& gate : gatesInRow)
      gatesInTile.insert(graph.gateId(gate));

    // check for gate ID
    for (const auto& gate : gatesInRow) {
      int gateId = -1;
      for (const auto [gatePtr, id] : graph.gateMap()) {
        if (gate == gatePtr.get()) {
          gateId = id;
          break; // found the gate in gateMap
        }
      }
      if (gateId == -1) {
        std::cerr << BOLDRED("Err: ") << "In checkConsistency_gateMemory(): "
          "Gate @ " << (void*)gate
          << " in row " << rowNumber << "is not found in gateMap.\n";
        return false;
      }
    }
  }
  // Check if all gates in gateMap are present in the tile
  if (gatesInTile.size() != graph.gateMap().size()) {
    std::cerr << BOLDRED("Err: ") << "In checkConsistency_gateMemory(): "
      "Mismatch in the sizes of gates in the tile and gateMap: "
      << gatesInTile.size() << " vs " << graph.gateMap().size() << "\n";
    return false;
  }
  return true;
}

bool CircuitGraphNode::checkConsistency() const {
  // Check all rows have the same size.
  return checkConsistency_rowSize(*this) && 
         checkConsistency_gateMap(*this) &&
         checkConsistency_gateMemory(*this);
}

void CircuitGraphNode::resizeRowsIfNeeded(int newNQubits) {
  if (newNQubits <= _nQubits)
    return;
  for (auto rowIt = tile_begin(), e = tile_end(); rowIt != e; ++rowIt)
    rowIt->resize(newNQubits);
  _nQubits = newNQubits;
}

void CircuitGraphNode::insertGate(QuantumGatePtr gate) {
  assert(gate != nullptr && "Inserting a null gate");
  resizeRowsIfNeeded(gate->qubits().back() + 1);
  // add gate to gateMap
  auto [gateMapIt, inserted] = _gateMap.insert({gate, _gateMapId++});
  assert(inserted && "Gate already exists");
  
  auto rowIt = --_tile.end();
  if (rowIt == _tile.end() || !isRowVacant(rowIt, *gate)) {
    // insert a new row if either tile is empty or the last row is not vacant
    rowIt = insertNewRow(tile_end());
  }

  for (auto q : gate->qubits()) {
    assert((*rowIt)[q] == nullptr && "Gate already exists");
    (*rowIt)[q] = gate.get();
  }
}

void CircuitGraphNode::insertGate(QuantumGatePtr gate, row_iterator rowIt) {
  assert(gate != nullptr && "Inserting a null gate");
  resizeRowsIfNeeded(gate->qubits().back() + 1);
  // add gate to gateMap
  auto [it, inserted] = _gateMap.insert({gate, _gateMapId++});
  assert(inserted && "Gate already exists");

  for (auto q : gate->qubits()) {
    assert((*rowIt)[q] == nullptr && "Gate already exists");
    (*rowIt)[q] = gate.get();
  }
}

void CircuitGraphNode::removeGate(row_iterator rowIt, int qubit) {
  assert(rowIt != _tile.end());
  assert(qubit >= 0 && qubit < _nQubits);
  auto gate = (*rowIt)[qubit];
  assert(gate != nullptr && "Removing a null gate");
  // erase from tile
  for (const auto& q : gate->qubits()) {
    assert((*rowIt)[q] == gate && "Gate not found in the row");
    (*rowIt)[q] = nullptr;
  }
  // erase from gateMap
  for (auto it = _gateMap.begin(), e = _gateMap.end(); it != e; ++it) {
    if (it->first.get() == gate) {
      _gateMap.erase(it);
      break;
    }
  }
}

bool CircuitGraphNode::isRowVacant(row_iterator rowIt,
                                   const QuantumGate& gate) const {
  assert(rowIt != _tile.end());
  for (auto q : gate.qubits()) {
    assert(q >= 0);
    if (q >= _nQubits)
      continue;
    if ((*rowIt)[q] != nullptr)
      return false;
  }
  return true;
}

void CircuitGraphNode::fuseAndInsertSameRow(
    row_iterator rowIt, int q0, int q1) {
  assert(rowIt != tile_end());
  auto* gate0 = (*rowIt)[q0];
  assert(gate0 != nullptr);
  auto* gate1 = (*rowIt)[q1];
  assert(gate1 != nullptr);
  assert(gate0 != gate1 && "Fusing the same gate");

  auto gateFused = cast::matmul(gate0, gate1);
  removeGate(rowIt, q0);
  removeGate(rowIt, q1);
  insertGate(gateFused, rowIt);
}

CircuitGraphNode::row_iterator
CircuitGraphNode::fuseAndInsertDiffRow(row_iterator rowItL, int qubit) {
  assert(rowItL != tile_end());
  auto* gateL = (*rowItL)[qubit];
  assert(gateL != nullptr);
  auto rowItR = std::next(rowItL);
  assert(rowItR != tile_end());
  auto* gateR = (*rowItR)[qubit];
  assert(gateR != nullptr);

  auto gateFused = cast::matmul(gateL, gateR);
  removeGate(rowItL, gateL->qubits()[0]);
  removeGate(rowItR, gateR->qubits()[0]);

  // prioritize iterR > iterL > between iterL and iterR
  if (isRowVacant(rowItR, *gateFused)) {
    insertGate(gateFused, rowItR);
    return rowItR;
  }
  if (isRowVacant(rowItL, *gateFused)) {
    insertGate(gateFused, rowItL);
    return rowItL;
  }
  auto rowItInserted = insertNewRow(rowItR);
  insertGate(gateFused, rowItInserted);
  return rowItInserted;
}

int CircuitGraphNode::gateId(const QuantumGate* gate) const {
  for (const auto& [itGate, id] : _gateMap) {
    if (itGate.get() == gate)
      return id;
  }
  return -1; // gate not found
}

QuantumGatePtr CircuitGraphNode::lookup(QuantumGate* gate) const {
  if (gate == nullptr)
    return nullptr;
  for (const auto& [itGate, id] : _gateMap) {
    if (itGate.get() == gate)
      return itGate;
  }
  return nullptr; // gate not found
}

void CircuitGraphNode::squeeze() {
  // first step: relocate gates to the top
  for (auto rowIt = ++tile_begin(), rowEnd = tile_end();
       rowIt != rowEnd;
       ++rowIt) {
    for (int q = 0; q < _nQubits; ++q) {
      auto* gate = (*rowIt)[q];
      if (gate == nullptr)
        continue;
      // find the top-most vacant row
      auto rowVacant = std::prev(rowIt);
      while (rowVacant != tile_end() && isRowVacant(rowVacant, *gate)) {
        --rowVacant;
      }
      ++rowVacant; // move to the next row, which is vacant
      if (rowVacant == rowIt)
        continue;
      // relocate the gate
      for (const auto& qq : gate->qubits()) {
        assert((*rowIt)[qq] == gate);
        assert((*rowVacant)[qq] == nullptr);
        (*rowIt)[qq] = nullptr;
        (*rowVacant)[qq] = gate;
      }
    }
  }

  // second step: remove empty rows
  auto rowIt = tile_end();
  auto rowBegin = tile_begin();
  while (rowIt != rowBegin) {
    --rowIt;
    bool isEmpty = true;
    for (int q = 0; q < _nQubits; ++q) {
      if ((*rowIt)[q] != nullptr) {
        isEmpty = false;
        break;
      }
    }
    if (!isEmpty) {
      // we are sure all gates are squeezed in step 1.
      // so terminate as soon as there is a non-empty row.
      break;
    }
    rowIt = _tile.erase(rowIt);
  }
}

static void push_back_if_not_in(
    std::vector<QuantumGate*>& gates, QuantumGate* gate) {
  if (gate != nullptr &&
      std::find(gates.begin(), gates.end(), gate) == gates.end()) {
    gates.push_back(gate);
  }
}

std::vector<QuantumGate*> CircuitGraphNode::getAllGates() const {
  std::vector<QuantumGate*> gates;
  auto rowEnd = tile_end();
  std::vector<QuantumGate*> rowGates;
  rowGates.reserve(_nQubits);
  for (auto rowIt = tile_begin(); rowIt != rowEnd; ++rowIt) {
    rowGates.clear();
    for (int q = 0; q < _nQubits; ++q)
      push_back_if_not_in(rowGates, (*rowIt)[q]);
    for (auto* gate : rowGates)
      gates.push_back(gate);
  }
  return gates;
}

static void push_back_if_not_in(
    std::vector<QuantumGatePtr>& gates, QuantumGatePtr gate) {
  if (gate != nullptr &&
      std::find(gates.begin(), gates.end(), gate) == gates.end()) {
    gates.push_back(gate);
  }
}

std::vector<QuantumGatePtr> CircuitGraphNode::getAllGatesShared() const {
  std::vector<QuantumGatePtr> gates;
  auto rowEnd = tile_end();
  std::vector<QuantumGatePtr> rowGates;
  rowGates.reserve(_nQubits);
  for (auto rowIt = tile_begin(); rowIt != rowEnd; ++rowIt) {
    rowGates.clear();
    for (int q = 0; q < _nQubits; ++q)
      push_back_if_not_in(rowGates, lookup((*rowIt)[q]));
    for (auto& gate : rowGates)
      gates.push_back(gate);
  }
  return gates;
}

std::ostream& CircuitGraphNode::print(std::ostream& os, int indent) const {
  return writeIndent(os, indent)
    << "cast.circuit_graph [@" << this
    << ", " << _gateMap.size() << " gates]\n";
}

std::ostream& CircuitGraphNode::displayInfo(std::ostream& os, int verbose) const {
  os << BOLDCYAN("=== Info of CircuitGraph @ " << this << " === ")
     << "(Verbose " << verbose << ")\n";
  os << CYAN("- nQubits:       ") << _nQubits << "\n";
  os << CYAN("- Num Gates:      ") << _gateMap.size() << "\n";
  os << CYAN("- Num Rows:       ") << _tile.size() << "\n";
  
  os << BOLDCYAN("====================================") << "\n";
  return os;
}

std::ostream& CircuitGraphNode::visualize(std::ostream& os, int verbose) const {
  os << BOLDCYAN("=== Visualizing CircuitGraph @ " << this << " === ")
     << "(Verbose " << verbose << ")\n";
  if (_tile.empty())
    return os << "<empty tile>\n";
  int width = static_cast<int>(std::log10(_gateMapId) + 1);
  if ((width & 1) == 0)
    width++;

  const std::string vbar =
      std::string(width / 2, ' ') + "|" + std::string(width / 2 + 1, ' ');

  for (const auto& row : _tile) {
    if (verbose > 1)
      os << &row << ": ";
    for (unsigned q = 0; q < nQubits(); q++) {
      if (const auto* gate = row[q]; gate != nullptr)
        os << std::setw(width) << std::setfill('0') << gateId(gate) << " ";
      else
        os << vbar;
    }
    os << "\n";
  }
  os << BOLDCYAN("====================================") << "\n";
  return os;
}