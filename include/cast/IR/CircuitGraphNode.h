#ifndef CAST_IR_CIRCUIT_GRAPH_NODE_H
#define CAST_IR_CIRCUIT_GRAPH_NODE_H

#include "cast/IR/IRNode.h"
#include "cast/QuantumGate.h"

#include <list>
#include <unordered_map>

namespace cast::ir {

class CircuitGraphNode {
private:
  int _rowSize;
  int _rowCapacity;
  using row_t = QuantumGate**;
  std::list<row_t> _tile;
  // struct TransparentHash {
  //   using is_transparent = void;
    
  //   std::size_t operator()(const QuantumGatePtr& gate) const noexcept {
  //     return std::hash<QuantumGate*>()(gate.get());
  //   }

  //   size_t operator()(const QuantumGate* gate) const noexcept {
  //     return std::hash<const QuantumGate*>()(gate);
  //   }
  // };

  // struct TransparentEqual {
  //   using is_transparent = void;

  //   bool operator()(const QuantumGatePtr& lhs, const QuantumGatePtr& rhs) const noexcept {
  //     return lhs.get() == rhs.get();
  //   }

  //   bool operator()(const QuantumGate* lhs, const QuantumGatePtr& rhs) const noexcept {
  //     return lhs == rhs.get();
  //   }

  //   bool operator()(const QuantumGatePtr& lhs, const QuantumGate* rhs) const noexcept {
  //     return lhs.get() == rhs;
  //   }
  // };

  // std::unordered_map<QuantumGatePtr, int, TransparentHash, TransparentEqual> _gateMap;

  /* TODO
   * We will need to lookup gates using raw pointers. Current approach is O(n)
   */

  // _gateMap manages memory and stores the id of gates
  std::unordered_map<QuantumGatePtr, int> _gateMap;
  using row_iterator = std::list<row_t>::iterator;
  using const_row_iterator = std::list<row_t>::const_iterator;

  void resizeRows(int size);
  void reserveRows(int capacity);
public:
  CircuitGraphNode(int desiredNQubits = 32)
    : _rowSize(desiredNQubits)
    , _rowCapacity(desiredNQubits)
    , _tile() {}

  CircuitGraphNode(const CircuitGraphNode&);
  CircuitGraphNode(CircuitGraphNode&&) noexcept;
  CircuitGraphNode& operator=(const CircuitGraphNode&);
  CircuitGraphNode& operator=(CircuitGraphNode&&) noexcept;

  ~CircuitGraphNode() {
    for (auto rowIt = _tile.begin(), e = _tile.end(); rowIt != e; ++rowIt)
      delete[] *rowIt;
  }

  void insertGate(QuantumGatePtr gate);
  void insertGate(QuantumGatePtr gate, row_iterator rowIt);

  void removeGate(row_iterator rowIt, int qubitIdx);

  std::ostream& print(std::ostream& os, int verbose=1) const;

  std::ostream& displayInfo(std::ostream& os, int verbose=1) const;

}; // class CircuitGraphNode
  
}; // namespace cast::ir


#endif // CAST_IR_CIRCUIT_GRAPH_NODE_H