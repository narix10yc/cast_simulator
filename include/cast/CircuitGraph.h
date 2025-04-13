#ifndef CAST_CIRCUITGRAPH_H
#define CAST_CIRCUITGRAPH_H

#include "cast/QuantumGate.h"
#include "cast/CircuitGraphContext.h"
#include "utils/List.h"

#include "llvm/ADT/SmallVector.h"

#include <array>
#include <vector>

namespace cast {

class CircuitGraph;

class GateNode {
public:
  friend class CircuitGraph;

  struct ConnectionInfo {
    int qubit;
    GateNode* lhsGate;
    GateNode* rhsGate;
  };

  int id;
  std::shared_ptr<QuantumGate> quantumGate;
  std::vector<ConnectionInfo> connections;

  GateNode(std::shared_ptr<QuantumGate> quantumGate, const CircuitGraph& graph);

  GateNode(const GateNode&) = delete;
  GateNode(GateNode&&) = delete;
  GateNode& operator=(const GateNode&) = delete;
  GateNode& operator=(GateNode&&) = delete;

  ~GateNode() = default;

  ConnectionInfo* findConnection(int qubit) {
    for (auto& item : connections) {
      if (item.qubit == qubit)
        return &item;
    }
    return nullptr;
  }

  const ConnectionInfo* findConnection(int qubit) const {
    for (const auto& item : connections) {
      if (item.qubit == qubit)
        return &item;
    }
    return nullptr;
  }

  void connect(GateNode* rhsGate, int q);
};

/// \brief A \c GateBlock is an aggregate of \c GateNode 's.
/// It is the primary layer for gate fusion. Aggregation relation is maintained
/// by the \c wires variable, which is internally a vector of qubit, left entry
/// node, and right entry node.
///
/// \c GateBlock can be `singleton`: when it contains exactly one \c GateNode .
/// In such cases, their respective \c quantumGate member will point to the same
/// \c QuantumGate object.
class GateBlock {
public:
  friend class CircuitGraph;

  struct WireInfo {
    int qubit;
    GateNode* lhsEntry;
    GateNode* rhsEntry;
  };

  int id;
  std::vector<WireInfo> wires;
  std::shared_ptr<QuantumGate> quantumGate;

  GateBlock();
  explicit GateBlock(GateNode* gateNode);

  GateBlock(const GateBlock&) = delete;
  GateBlock(GateBlock&&) = delete;
  GateBlock& operator=(const GateBlock&) = delete;
  GateBlock& operator=(GateBlock&&) = delete;

  ~GateBlock() = default;

  std::ostream& displayInfo(std::ostream& os) const;

  std::vector<GateNode*> getOrderedGates() const;

  bool isSingleton() const;

  size_t countGates() const { return getOrderedGates().size(); }

  int nQubits() const { return wires.size(); }

  WireInfo* findWire(int qubit) {
    for (auto& wire : wires) {
      if (wire.qubit == qubit)
        return &wire;
    }
    return nullptr;
  }

  const WireInfo* findWire(int qubit) const {
    for (const auto& wire : wires) {
      if (wire.qubit == qubit)
        return &wire;
    }
    return nullptr;
  }

  bool hasSameTargets(const GateBlock& other) const {
    if (nQubits() != other.nQubits())
      return false;
    for (const auto& data : other.wires) {
      if (findWire(data.qubit) == nullptr)
        return false;
    }
    return true;
  }

  void internalFuse() { assert(false && "Not Implemented"); }
};

/// TODO: Refactor to use std::list
class CircuitGraph {
private:
  std::unique_ptr<CircuitGraphContext> _context;
public:
  using row_t = std::array<GateBlock*, 42>;
  using tile_t = utils::List<row_t>;
  using list_node_t = tile_t::Node;

  // iterators
  using tile_iter_t = tile_t::iterator;
  // using citer_t = tile_t::const_iterator;
  // using riter_t = tile_t::reverse_iterator;
  // using criter_t = tile_t::const_reverse_iterator;

private:
  tile_t _tile;
public:
  int nQubits;

  CircuitGraph()
  : _context(std::make_unique<CircuitGraphContext>()), _tile(), nQubits(0) {
    _tile.emplace_back();
  }

  CircuitGraph(const CircuitGraph&) = delete;
  CircuitGraph(CircuitGraph&&) = delete;
  CircuitGraph& operator=(const CircuitGraph&) = delete;
  CircuitGraph& operator=(CircuitGraph&&) = delete;

  static void QFTCircuit(int nQubits, CircuitGraph& graph);
  static CircuitGraph ALACircuit(int nQubits, int nrounds);

  static CircuitGraph GetTestCircuit(
    const GateMatrix& gateMatrix, int nQubits, int nrounds);

  tile_t& tile() { return _tile; }
  const tile_t& tile() const { return _tile; }

  tile_iter_t tile_begin() { return _tile.begin(); }
  tile_iter_t tile_end() { return _tile.end(); }

  void clear();

  // CircuitGraphContext& getContext() { return _context; }

  template<typename... Args>
  GateNode* acquireGateNodeForward(Args&&... args) {
    return _context->gateNodePool.acquire(std::forward<Args>(args)...);
  }

  template<typename... Args>
  GateBlock* acquireGateBlockForward(Args&&... args) {
    return _context->gateBlockPool.acquire(std::forward<Args>(args)...);
  }

  /// Acquire a \c GateBlock by fusing two blocks together. Notice that
  /// \c lhsBlock is applied `before` \c rhsBlock.
  GateBlock* acquireGateBlock(GateBlock* lhsBlock, GateBlock* rhsBlock);

  void releaseGateNode(GateNode* gateNode) const {
    assert(gateNode != nullptr && "Releasing null gateNode");
    _context->gateNodePool.release(gateNode);
  }

  /// Release the memory of the \c GateBlock
  void releaseGateBlock(GateBlock* gateBlock) const {
    assert(gateBlock != nullptr && "Releasing null gateBlock");
    _context->gateBlockPool.release(gateBlock);
  }

  /// Append a quantum gate to the tile. Quantum gate must be managed by
  /// \c *this
  void appendGate(std::shared_ptr<QuantumGate> quantumGate);

  /// @brief Erase empty rows in the tile
  void eraseEmptyRows();

  bool isRowVacant(const row_t& row, const GateBlock* block) const {
    for (const auto& q : block->quantumGate->qubits)
      if (row[q] != nullptr)
        return false;
    return true;
  }

  /// Update connections of blocks. By assumption nodes connections are always
  /// preserved. In Debug mode this function will check for node connections.
  void updateBlockConnections(tile_iter_t it, int q);

  list_node_t* repositionBlockUpward(list_node_t* ln, int q);
  tile_iter_t repositionBlockUpward(tile_iter_t it, int q) {
    return tile_iter_t(repositionBlockUpward(it.raw_ptr(), q));
  }

  list_node_t* repositionBlockDownward(list_node_t* ln, int q);
  tile_iter_t repositionBlockDownward(tile_iter_t it, int q) {
    return tile_iter_t(repositionBlockDownward(it.raw_ptr(), q));
  }

  // Squeeze graph and make it compact
  void squeeze();

  /// @brief Try to insert block to a specified row. Three outcome may happen:
  /// - If \p it is vacant, insert \p block there. Otherwise,
  /// - If \p it+1 is vacant, insert \p block there. Otherwise,
  /// - Insert a separate row between \p it and \p it+1 and place \p block
  ///   there.
  tile_iter_t insertBlock(tile_iter_t it, GateBlock* block);

  /// @brief Get the number of blocks with each size.
  /// @return ret[i] is the number of blocks with size i. Therefore, ret[0] is
  /// always 0, and ret.size() == largest_size + 1.
  std::vector<int> getBlockSizes() const;

  /// @return ordered vector of blocks
  std::vector<GateBlock*> getAllBlocks() const;

  // std::vector<std::vector<int>> getBlockOpCountHistogram() const;

  size_t countBlocks() const { return getAllBlocks().size(); }

  // size_t countGates() const {
  //   const auto allBlocks = getAllBlocks();
  //   size_t sum = 0;
  //   for (const auto& block : allBlocks)
  //     sum += block->countGates();
  //   return sum;
  // }
  //
  // size_t countTotalOps() const {
  //   const auto allBlocks = getAllBlocks();
  //   size_t sum = 0;
  //   for (const auto& block : allBlocks) {
  //     assert(block->quantumGate != nullptr);
  //     sum += block->quantumGate->opCount();
  //   }
  //   return sum;
  // }
  //
  // void relabelBlocks() const;

  /// @brief Console print the tile.
  /// @param verbose If > 1, also print the address of each row in front
  std::ostream& print(std::ostream& os = std::cerr, int verbose = 1) const;

  std::ostream& displayInfo(
    std::ostream& os = std::cerr, int verbose = 1) const;

  void deepCopy(CircuitGraph& other) const;
};

}; // namespace cast

#endif // CAST_CIRCUITGRAPH_H