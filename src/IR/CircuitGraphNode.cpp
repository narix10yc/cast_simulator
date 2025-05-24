#include "cast/IR/IRNode.h"
#include "utils/iocolor.h"

#include <set>

using namespace cast::ir;

int CircuitGraphNode::_gateMapId = 0;

// checks if all rows in the circuit graph have the same size
static bool checkConsistency_rowSize(const CircuitGraphNode& graph) {
  auto it = graph.tile_begin();
  const auto end = graph.tile_end();
  if (it == end)
    return true; // empty graph is consistent
  for (; it != end; ++it) {
    if (it->capacity() != graph.rowCapacity()) {
      std::cerr << BOLDRED("Err: ") << "In checkConsistency_rowSize(): "
        "Inconsistent row capacity: graph capacity is " << graph.rowCapacity()
        << ", but row capacity is " << it->capacity() << "\n";
      return false;
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
      gatesInTile.insert(graph.gateId(*gate));

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

void CircuitGraphNode::reserveRows(int capacity) {
  if (capacity <= _rowCapacity)
    return;
  _rowCapacity = capacity;
  for (auto rowIt = _tile.begin(), e = _tile.end(); rowIt != e; ++rowIt)
    rowIt->reserve(capacity);
}

void CircuitGraphNode::insertGate(QuantumGatePtr gate) {
  assert(gate != nullptr);
  reserveRows(gate->qubits().back() + 1);
  // add gate to gateMap
  auto [gateMapIt, inserted] = _gateMap.insert({gate, _gateMapId++});
  assert(inserted && "Gate already exists");
  
  auto rowIt = --_tile.end();
  if (rowIt == _tile.end() || !isRowVacant(rowIt, *gate)) {
    // insert a new row if either tile is empty or the last row is not vacant
    rowIt = _tile.insert(_tile.end(), row_t(_rowCapacity, nullptr));
  }

  for (auto q : gate->qubits()) {
    assert((*rowIt)[q] == nullptr && "Gate already exists");
    (*rowIt)[q] = gate.get();
  }
}

void CircuitGraphNode::insertGate(QuantumGatePtr gate, row_iterator rowIt) {
  assert(gate != nullptr);
  reserveRows(gate->qubits().back() + 1);
  // add gate to gateMap
  auto [it, inserted] = _gateMap.insert({gate, _gateMapId++});
  assert(inserted && "Gate already exists");

  for (auto q : gate->qubits()) {
    assert((*rowIt)[q] == nullptr && "Gate already exists");
    (*rowIt)[q] = gate.get();
  }
}

void CircuitGraphNode::removeGate(row_iterator rowIt, int qubitIdx) {
  assert(rowIt != _tile.end());
  assert(qubitIdx >= 0 && qubitIdx < _nQubits);
  auto gate = (*rowIt)[qubitIdx];
  if (gate == nullptr)
    return;
  (*rowIt)[qubitIdx] = nullptr;
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
  assert(gate.qubits().back() < _rowCapacity);
  for (auto q : gate.qubits()) {
    if ((*rowIt)[q] != nullptr)
      return false;
  }
  return true;
}

int CircuitGraphNode::gateId(const QuantumGate& gate) const {
  for (const auto& [itGate, id] : _gateMap) {
    if (itGate.get() == &gate)
      return id;
  }
  return -1; // gate not found
}

std::ostream& CircuitGraphNode::print(std::ostream& os, int indent) const {
  return writeIndent(os, indent)
    << "cast.circuit_graph [@" << this
    << ", " << _gateMap.size() << " gates]\n";
}

std::ostream& CircuitGraphNode::displayInfo(std::ostream& os, int verbose) const {
  os << BOLDCYAN("=== Info of CircuitGraph @ " << this << " === ")
     << "(Verbose " << verbose << ")\n";
  os << CYAN("- Num Gates:      ") << _gateMap.size() << "\n";
  os << CYAN("- Num Rows:       ") << _tile.size() << "\n";
  os << CYAN("- rowCapacity:    ") << _rowCapacity << "\n";
  
  os << BOLDCYAN("====================================") << "\n";
  return os;
}

std::ostream& CircuitGraphNode::visualize(std::ostream& os, int verbose) const {
  os << BOLDCYAN("=== Visualizing CircuitGraph @ " << this << " === ")
     << "(Verbose " << verbose << ")\n";
  
  for (const auto& row : _tile) {
    for (const auto& gate : row) {
      if (gate != nullptr) {
        os << gateId(*gate) << " ";
      } else {
        os << ". ";
      }
    }
    os << "\n";
  }
  
  os << BOLDCYAN("====================================") << "\n";
  return os;
}