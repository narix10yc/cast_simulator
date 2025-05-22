#ifndef CAST_TRANSFORM_TRANSFORM_H
#define CAST_TRANSFORM_TRANSFORM_H

#include "cast/IR/IRNode.h"
#include "new_parser/AST.h"
#include <memory>

namespace cast {

namespace transform {

/// @brief Convert a GateApplyStmt to a GateMatrixPtr.
GateMatrixPtr convertGate(cast::draft::ast::GateApplyStmt* astGate,
                          cast::draft::ast::ASTContext& astCtx);

                          

std::unique_ptr<ir::CircuitNode> convertCircuit(
    const draft::ast::CircuitStmt& astCircuit, draft::ast::ASTContext& astCtx);

} // namespace transform

} // namespace cast;

#endif // CAST_TRANSFORM_TRANSFORM_H