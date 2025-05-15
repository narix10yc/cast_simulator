#include "cast/CircuitGraph.h"

using namespace cast;

void CircuitGraph::resizeRows(int size) {
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

void CircuitGraph::reserveRows(int capacity) {
  if (capacity <= _rowCapacity)
    return;
  for (auto rowIt = _tile.begin(), e = _tile.end(); rowIt != e; ++rowIt) {
    auto* newRow = new QuantumGatePtr[capacity];
    for (int i = 0; i < _rowSize; ++i)
      newRow[i] = std::move((*rowIt)[i]);
    delete[] *rowIt;
    *rowIt = newRow;
  }
  _rowCapacity = capacity;
}

void CircuitGraph::addGate(QuantumGatePtr gate) {
  assert(gate != nullptr);
  int nQubits = gate->nQubits();
  if (nQubits > _rowSize)
    resizeRows(nQubits);
  if (nQubits > _rowCapacity)
    reserveRows(nQubits);

  auto rowIt = _tile.begin();
  for (int i = 0; i < nQubits; ++i, ++rowIt) {
    (*rowIt)[i] = gate;
  }
}