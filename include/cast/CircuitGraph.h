#ifndef CAST_CIRCUIT_GRAPH_H
#define CAST_CIRCUIT_GRAPH_H

#include "utils/ObjectPool.h"

namespace cast {

class CircuitGraphNode {

}; // class CircuitGraphNode

class CircuitGraph {
private:
  utils::ObjectPool<CircuitGraphNode> _nodePool;


public:
  CircuitGraph() = default;
  ~CircuitGraph() = default;

  CircuitGraph(const CircuitGraph&) = delete;
  CircuitGraph& operator=(const CircuitGraph&) = delete;
  CircuitGraph(CircuitGraph&&) = delete;
  CircuitGraph& operator=(CircuitGraph&&) = delete;


}; // class CircuitGraph
  
}; // namespace cast

#endif // CAST_CIRCUIT_GRAPH_H