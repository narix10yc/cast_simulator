#ifndef CAST_TRANSFORM_ASTCIRCUIT2IRCIRCUIT_H
#define CAST_TRANSFORM_ASTCIRCUIT2IRCIRCUIT_H

#include "cast/IR/IRNode.h"
#include "new_parser/AST.h"
#include <memory>

namespace cast {

namespace transform {

std::unique_ptr<ir::CircuitNode> ASTCircuit2IRCircuit(
    const draft::ast::CircuitStmt& astCircuit, draft::ast::ASTContext& astCtx);

} // namespace transform

} // namespace cast;

#endif // CAST_TRANSFORM_ASTCIRCUIT2IRCIRCUIT_H