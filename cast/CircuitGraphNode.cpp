#include "cast/IR/CircuitGraphNode.h"
#include "utils/iocolor.h"

using namespace cast::ir;

CircuitGraphNode::CircuitGraphNode(const CircuitGraphNode& G) {
  _rowSize = G._rowSize;
  _rowCapacity = G._rowCapacity;
  for (auto rowIt = G._tile.begin(), e = G._tile.end(); rowIt != e; ++rowIt) {
    auto* newRow = new QuantumGate*[_rowCapacity];
    std::memcpy(newRow, *rowIt, _rowSize * sizeof(QuantumGate*));
    _tile.push_back(newRow);
  }
  _gateMap = G._gateMap;
}

CircuitGraphNode::CircuitGraphNode(CircuitGraphNode&& G) noexcept
  : _rowSize(G._rowSize)
  , _rowCapacity(G._rowCapacity)
  , _tile(std::move(G._tile))
  , _gateMap(std::move(G._gateMap)) {
  G._rowSize = 0;
  G._rowCapacity = 0;
}

CircuitGraphNode& CircuitGraphNode::operator=(const CircuitGraphNode& G) {
  if (this == &G)
    return *this;
  this->~CircuitGraphNode();
  new (this) CircuitGraphNode(G);
  return *this;
}

CircuitGraphNode& CircuitGraphNode::operator=(CircuitGraphNode&& G) noexcept {
  if (this == &G)
    return *this;
  this->~CircuitGraphNode();
  new (this) CircuitGraphNode(std::move(G));
  return *this;
}

void CircuitGraphNode::resizeRows(int size) {
  assert(size > 0);
  if (size == _rowSize)
    return;
  if (size < _rowSize) {
    for (auto rowIt = _tile.begin(), e = _tile.end(); rowIt != e; ++rowIt) {
      for (int i = size; i < _rowSize; ++i)
        (*rowIt)[i] = nullptr;
    }
    _rowSize = size;
    return;
  }

  // After resizing, size equals to capacity
  reserveRows(size);
  _rowSize = size;
}

void CircuitGraphNode::reserveRows(int capacity) {
  if (capacity <= _rowCapacity)
    return;
  for (auto rowIt = _tile.begin(), e = _tile.end(); rowIt != e; ++rowIt) {
    auto* newRow = new QuantumGate*[capacity];
    std::memcpy(newRow, *rowIt, _rowSize * sizeof(QuantumGate*));
    delete[] *rowIt;
    *rowIt = newRow;
  }
  _rowCapacity = capacity;
}

void CircuitGraphNode::insertGate(QuantumGatePtr gate) {
  assert(gate != nullptr);
  resizeRows(gate->qubits().back() + 1);
  // add gate to gateMap
  auto [it, inserted] = _gateMap.insert({gate, _gateMap.size()});
  assert(inserted && "Gate already exists");
  
  bool vacant = true;
  if (_tile.empty()) {
    vacant = false;
  }
  else {
    auto lastRow = _tile.back();
    for (auto q : gate->qubits()) {
      if (lastRow[q] != nullptr) {
        vacant = false;
        break;
      }
    }
  }
  QuantumGate** row;
  if (vacant) {
    row = new QuantumGate*[_rowCapacity];
    std::memset(row, 0, _rowCapacity * sizeof(QuantumGate*));
    _tile.push_back(row);
  } else {
    row = _tile.back();
  }
  
  for (auto q : gate->qubits()) {
    assert(row[q] == nullptr && "Gate already exists");
    row[q] = gate.get();
  }
}

void CircuitGraphNode::insertGate(QuantumGatePtr gate, row_iterator rowIt) {
  assert(gate != nullptr);
  resizeRows(gate->qubits().back() + 1);
  // add gate to gateMap
  auto [it, inserted] = _gateMap.insert({gate, _gateMap.size()});
  assert(inserted && "Gate already exists");

  for (auto q : gate->qubits()) {
    assert((*rowIt)[q] == nullptr && "Gate already exists");
    (*rowIt)[q] = gate.get();
  }
}

void CircuitGraphNode::removeGate(row_iterator rowIt, int qubitIdx) {
  assert(rowIt != _tile.end());
  assert(qubitIdx >= 0 && qubitIdx < _rowSize);
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

std::ostream& CircuitGraphNode::print(std::ostream& os, int verbose) const {
  return os;
}

std::ostream& CircuitGraphNode::displayInfo(std::ostream& os, int verbose) const {
  os << BOLDCYAN("=== Info of CircuitGraph @ " << this << " === ")
     << "(Verbose " << verbose << ")\n";
  os << CYAN("- rowSize: ") << _rowSize << "\n";
  os << CYAN("- rowCapacity: ") << _rowCapacity << "\n";
  return os;
}