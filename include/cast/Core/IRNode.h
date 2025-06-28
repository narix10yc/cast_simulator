#ifndef CAST_CORE_IRNODE_H
#define CAST_CORE_IRNODE_H

#include "cast/Core/QuantumGate.h"
#include "utils/MaybeError.h"
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
protected:
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

  // The implementation of visualize(). visualize() function is only provided
  // for class CircuitGraphNode and CircuitNode.
  // Use parameter name n_qubits to avoid name conflicts.
  virtual std::ostream&
  impl_visualize(std::ostream& os, int width, int n_qubits) const {
    assert(false && "Not implemented yet (called from base class)");
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

  void push_front(std::unique_ptr<IRNode> node) {
    nodes.insert(nodes.begin(), std::move(node));
  }
  
  void push_back(std::unique_ptr<IRNode> node) {
    nodes.push_back(std::move(node));
  }

  size_t size() const { return nodes.size(); }

  bool empty() const { return nodes.empty(); }

  std::ostream& print(std::ostream& os, int indent) const override;

  std::ostream&
  impl_visualize(std::ostream& os, int width, int n_qubits) const override;
  
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

  std::ostream&
  impl_visualize(std::ostream& os, int width, int n_qubits) const override;

  static bool classof(const IRNode* node) {
    return node->getKind() == IRNode_IfMeasure;
  }
}; // class IfMeasureNode

/// We use an unordered_map \c _gateMap memory management and index tracking.
/// When inserting gates, always use \c insertGate .
/// Correspondingly, when removing gates, always use \c removeGate .
class CircuitGraphNode : public IRNode {
private:
  // _nQubits always equal to the size of each row
  int _nQubits;
  using row_t = std::vector<QuantumGate*>;
  std::list<row_t> _tile;
public:
  static int _gateMapId;

  // This is the width to display in each qubit wire.
  static int getWidthForVisualize();
  
  static constexpr int DefaultRowCapacity = 32;

  using row_iterator = std::list<row_t>::iterator;
  using const_row_iterator = std::list<row_t>::const_iterator;
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
   * We sometimes look up gates using raw pointers. Current approach is O(n)
   */

private:
  // _gateMap manages memory and stores the id of gates
  std::unordered_map<QuantumGatePtr, int> _gateMap;

  void resizeRowsIfNeeded(int size);
public:
  /** TODO: a bit awkward to expose the following functions used in fusion
   * as public.
   **/

  // This simply calls _tile.emplace(rowIt, row_t(DefaultRowCapacity, nullptr));
  row_iterator insertNewRow(row_iterator rowIt) {
    return _tile.emplace(rowIt, row_t(DefaultRowCapacity, nullptr));
  }

  /// Fuse two gates on the same row given by (*rowIt)[q0] and (*rowIt)[q1].
  /// This function removes old gates from the graph and inserts the fused gate.
  void fuseAndInsertSameRow(row_iterator rowIt, int q0, int q1);

  /// Fuse two gates on different rows given by (*rowItL)[qubit] and
  /// (*std::next(rowItL))[qubit].
  /// This function removes old gates from the graph and inserts the fused gate.
  /// @return the tile iterator of the fused gate
  row_iterator fuseAndInsertDiffRow(row_iterator rowItL, int qubit);
public:
  CircuitGraphNode()
    : IRNode(IRNode_CircuitGraph)
    , _nQubits(0)
    , _tile() {}

  // Recommend to use assert(checkConsistency()) 
  bool checkConsistency() const;

  /// @brief Insert a gate into the end of the circuit graph.
  row_iterator insertGate(QuantumGatePtr gate);

  /// @brief Insert a gate into the circuit graph at or before rowIt.
  /// @return The row in which the gate is inserted.
  row_iterator insertGate(QuantumGatePtr gate, row_iterator rowIt);

  /// @brief Remove a gate from the circuit graph at a specific row and qubit.
  void removeGate(row_iterator rowIt, int qubit);

  bool isRowVacant(row_iterator rowIt,
                   const QuantumGate::TargetQubitsType& qubits) const;
  
  /// @brief Remove the two gates \c (*rowItL)[qubit] and
  /// \c (*std::next(rowItL))[qubit], and insert gate \c gate into the tile.
  /// Often this function is called when \c gate is the fused gate of the 
  /// two removed gates.
  /// The row that \c gate is inserted into takes the following priority:
  /// 1. Put \c gate in row \c std::next(rowItL) if is vacant; If not,
  /// 2. Put \c gate in row \c rowItL if is vacant. If not,
  /// 3. Insert a new row immediately after \c rowItL and put \c gate there.
  /// @return row of the inserted gate
  row_iterator replaceGatesOnConsecutiveRowsWith(
      QuantumGatePtr gate, row_iterator rowItL, int qubit);

  /// @brief Squeeze the circuit graph between rows [rowIt, tile_end()).
  void squeeze(row_iterator beginIt);

  void squeeze() { return squeeze(tile_begin()); }

  const std::unordered_map<QuantumGatePtr, int>& gateMap() const {
    return _gateMap;
  }

  // Return the id of the gate in the circuit graph. Return -1 if gate is not
  // in this circuit graph.
  int gateId(const QuantumGate* gate) const;

  int gateId(QuantumGatePtr gate) const {
    auto it = _gateMap.find(gate);
    if (it != _gateMap.end())
      return it->second;
    return -1; // gate not found
  }

  // Look up a gate in the graph. Return nullptr if not found.
  QuantumGatePtr lookup(QuantumGate* gate) const;

  std::list<row_t>& tile() { return _tile; }
  const std::list<row_t>& tile() const { return _tile; }

  row_iterator tile_begin() { return _tile.begin(); }
  const_row_iterator tile_begin() const { return _tile.begin(); }

  row_iterator tile_end() { return _tile.end(); }
  const_row_iterator tile_end() const { return _tile.end(); }

  int nQubits() const { return _nQubits; }

  size_t nGates() const { return _gateMap.size(); }

  // Collect gates in the circuit graph in order. This methods returns a vector
  // of raw pointers. These pointers could be invalidated when the circuit graph
  // goes out of scope. Use \c getAllGatesShared() to retain gate memories. 
  std::vector<QuantumGate*> getAllGates() const;

  // Collect gates in the circuit graph in order. 
  std::vector<QuantumGatePtr> getAllGatesShared() const;

  std::ostream& print(std::ostream& os, int indent) const override;

  std::ostream& displayInfo(std::ostream& os, int verbose=1) const;

  std::ostream& visualize(std::ostream& os) const;

  std::ostream&
  impl_visualize(std::ostream& os, int width, int n_qubits) const override;

  static bool classof(const IRNode* node) {
    return node->getKind() == IRNode_CircuitGraph;
  }

}; // class CircuitGraphNode

class CircuitNode : public IRNode {
public:
  std::string name;
  CompoundNode body;

  CircuitNode(const std::string& name)
    : IRNode(IRNode_Circuit), name(name), body() {}

  std::ostream& print(std::ostream& os, int indent) const override;

  // Grab all circuit graphs inside this circuit. Notice that when `this` goes
  // out of scope, the returned pointers will be invalidated.
  std::vector<CircuitGraphNode*> getAllCircuitGraphs() const;

  unsigned countNumCircuitGraphs() const;

  std::ostream& displayInfo(std::ostream& os, int verbose=1) const;

  std::ostream& visualize(std::ostream& os) const;

  std::ostream&
  impl_visualize(std::ostream& os, int width, int n_qubits) const override;

  static bool classof(const IRNode* node) {
    return node->getKind() == IRNode_Circuit;
  }
}; // class CircuitNode

} // namespace cast::ir

namespace cast {
  /// @brief Parse a QASM file and return a cast::ir::CircuitNode.
  /// The definition is in src/Core/IR/ParseCircuitFromQASM.cpp.
  cast::MaybeError<ir::CircuitNode>
  parseCircuitFromQASMFile(const std::string& fileName);
}; // namespace cast

#endif // CAST_CORE_IRNODE_H