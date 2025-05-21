#ifndef CAST_TRANSFORM_ASTCIRCUIT2IRCIRCUIT_H
#define CAST_TRANSFORM_ASTCIRCUIT2IRCIRCUIT_H

namespace cast {

namespace draft {
  class ASTContext;
  class CircuitStmt;
}

namespace ir {
  class CircuitNode;
}

namespace transform {

ir::CircuitNode* ASTCircuit2IRCircuit(const draft::CircuitStmt& astCircuit,
                                      draft::ASTContext& astCtx);

} // namespace transform

} // namespace cast;

#endif // CAST_TRANSFORM_ASTCIRCUIT2IRCIRCUIT_H