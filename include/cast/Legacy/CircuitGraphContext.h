#ifndef CAST_LEGACY_CIRCUITGRAPH_CONTEXT_H
#define CAST_LEGACY_CIRCUITGRAPH_CONTEXT_H

#include "utils/ObjectPool.h"

namespace cast::legacy {

class GateNode;
class GateBlock;

class CircuitGraphContext {
public:
  static int GateNodeCount;
  static int GateBlockCount;

  /// Memory management
  utils::ObjectPool<GateNode> gateNodePool;
  utils::ObjectPool<GateBlock> gateBlockPool;

  CircuitGraphContext() = default;
};

} // namespace cast

#endif // CAST_LEGACY_CIRCUITGRAPH_CONTEXT_H
