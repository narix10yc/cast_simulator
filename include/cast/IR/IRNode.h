#ifndef CAST_IR_IRNODE_H
#define CAST_IR_IRNODE_H

#include "cast/QuantumGate.h"
#include <iostream>
#include <list>
#include <unordered_map>

namespace cast::ir {

/// @brief Base class for all CAST IR nodes.
class IRNode {
public:
  /// The LLVM-style RTTI. Indentation corresponds to class hirearchy.
  enum NodeKind {
    IRNode_Base,
      IRNode_Compound,
      IRNode_IfMeasure,
      IRNode_Circuit,
      IRNode_CircuitGraph,
      IRNode_OutMeasure,
      IRNode_OutState,
      IRNode_End
  };
private:
  NodeKind _kind;
public:
  explicit IRNode(NodeKind kind) : _kind(kind) {}

  NodeKind getKind() const { return _kind; }

  virtual ~IRNode() = default;

  std::ostream& writeIndent(std::ostream& os, int indent) const {
    assert(indent >= 0);
    for (int i = 0; i < indent; ++i)
      os.put(' ');
    return os;
  }

  virtual std::ostream& print(std::ostream& os, int indent) const {
    return os << "IRNode @ " << this;
  }
}; // class IRNode

/// @brief A wrapper of std::vector<std::unique_ptr<IRNode>>.
class CompoundNode : public IRNode {
public:
  std::vector<std::unique_ptr<IRNode>> nodes;
  CompoundNode() : IRNode(IRNode_Compound), nodes() {}

  void push_back(std::unique_ptr<IRNode> node) {
    nodes.push_back(std::move(node));
  }

  size_t size() const { return nodes.size(); }

  std::ostream& print(std::ostream& os, int indent) const override;

  static bool classof(const IRNode* node) {
    return node->getKind() == IRNode_Compound;
  }
}; // class CompoundNode

class IfMeasureNode : public IRNode {
public:
  int qubit;
  CompoundNode thenBody;
  CompoundNode elseBody;

  IfMeasureNode(int qubit)
    : IRNode(IRNode_IfMeasure), qubit(qubit), thenBody(), elseBody() {}

  std::ostream& print(std::ostream& os, int indent) const override;

  static bool classof(const IRNode* node) {
    return node->getKind() == IRNode_IfMeasure;
  }
}; // class IfMeasureNode

class CircuitNode : public IRNode {
public:
  std::string name;
  CompoundNode body;

  CircuitNode(const std::string& name)
    : IRNode(IRNode_Circuit), name(name), body() {}

  std::ostream& print(std::ostream& os, int indent) const override;

  unsigned countNumCircuitGraphs() const;

  std::ostream& displayInfo(std::ostream& os, int verbose=1) const;

  static bool classof(const IRNode* node) {
    return node->getKind() == IRNode_Circuit;
  }
}; // class CircuitNode

class CircuitGraphNode : public IRNode {
private:
  int _rowSize;
  int _rowCapacity;
  using row_t = std::vector<QuantumGate*>;
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
   * We prefer to look up gates using raw pointers. Current approach is O(n)
   */

  // _gateMap manages memory and stores the id of gates
  std::unordered_map<QuantumGatePtr, int> _gateMap;
  using row_iterator = std::list<row_t>::iterator;
  using const_row_iterator = std::list<row_t>::const_iterator;

  void reserveRows(int capacity);
public:
  CircuitGraphNode(int desiredNQubits = 32)
    : IRNode(IRNode_CircuitGraph)
    , _rowSize(desiredNQubits)
    , _rowCapacity(desiredNQubits)
    , _tile() {}

  /// Use this method to add gates into the circuit graph. It will be stored in
  /// an unordered_map \c _gateMap for memory management and index tracking.
  /// Correspondingly, when removing gates, use \c removeGate and pass in which
  /// row and qubit index.
  void insertGate(QuantumGatePtr gate);
  void insertGate(QuantumGatePtr gate, row_iterator rowIt);

  void removeGate(row_iterator rowIt, int qubitIdx);

  bool isRowVacant(row_iterator rowIt, const QuantumGate& gate) const;

  size_t nGates() const { return _gateMap.size(); }

  std::ostream& print(std::ostream& os, int indent) const override;

  std::ostream& displayInfo(std::ostream& os, int verbose=1) const;

  static bool classof(const IRNode* node) {
    return node->getKind() == IRNode_CircuitGraph;
  }

}; // class CircuitGraphNode

} // namespace cast::ir

#endif // CAST_IR_IRNODE_H