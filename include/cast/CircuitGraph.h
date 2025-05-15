#ifndef CAST_CIRCUIT_GRAPH_H
#define CAST_CIRCUIT_GRAPH_H

#include "cast/QuantumGate.h"
#include <list>

namespace cast {

class CircuitGraph {
private:
  int _rowSize;
  int _rowCapacity;
  std::list<QuantumGatePtr*> _tile;

  using row_iterator = std::list<QuantumGatePtr*>::iterator;
  using const_row_iterator = std::list<QuantumGatePtr*>::const_iterator;

  void resizeRows(int size);
  void reserveRows(int capacity);
public:
  CircuitGraph(int desiredNQubits = 32)
    : _rowSize(desiredNQubits)
    , _rowCapacity(desiredNQubits)
    , _tile() {}

  CircuitGraph(const CircuitGraph&) = delete;
  CircuitGraph& operator=(const CircuitGraph&) = delete;
  CircuitGraph(CircuitGraph&&) = delete;
  CircuitGraph& operator=(CircuitGraph&&) = delete;

  ~CircuitGraph() {
    for (auto rowIt = _tile.begin(), e = _tile.end(); rowIt != e; ++rowIt)
      delete[] *rowIt;
  }

  void addGate(QuantumGatePtr gate);


}; // class CircuitGraph
  
}; // namespace cast


#endif // CAST_CIRCUIT_GRAPH_H