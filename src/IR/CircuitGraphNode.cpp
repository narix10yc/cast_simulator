#include "cast/IR/IRNode.h"
#include "utils/iocolor.h"

using namespace cast::ir;

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
  auto [gateMapIt, inserted] = _gateMap.insert({gate, _gateMap.size()});
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

std::ostream& CircuitGraphNode::print(std::ostream& os, int indent) const {
  return writeIndent(os, indent)
    << "cast.circuit_graph [@" << this
    << ", " << _gateMap.size() << " gates]\n";
}

std::ostream& CircuitGraphNode::displayInfo(std::ostream& os, int verbose) const {
  os << BOLDCYAN("=== Info of CircuitGraph @ " << this << " === ")
     << "(Verbose " << verbose << ")\n";
  os << CYAN("- nGates:      ") << _gateMap.size() << "\n";
  os << CYAN("- rowCapacity: ") << _rowCapacity << "\n";
  return os;
}