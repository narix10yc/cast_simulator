#ifndef CAST_TRANSFORM_TRANSFORM_H
#define CAST_TRANSFORM_TRANSFORM_H

#include "cast/Core/AST/AST.h"
#include "cast/Core/AST/ASTContext.h"
#include "cast/Core/IRNode.h"
#include "openqasm/AST.h"

namespace cast {

namespace transform {

cast::ast::CircuitStmt*
cvtQasmCircuitToAstCircuit(const openqasm::ast::RootNode& qasmRoot,
                           ast::ASTContext& astCtx);

/// @brief Convert a GateApplyStmt to a GateMatrixPtr.
GateMatrixPtr cvtAstGateToGateMatrix(cast::ast::GateApplyStmt* astGate,
                                     cast::ast::ASTContext& astCtx);

std::unique_ptr<cast::ir::CircuitNode>
cvtAstCircuitToIrCircuit(const cast::ast::CircuitStmt& astCircuit,
                         cast::ast::ASTContext& astCtx);

} // namespace transform

} // namespace cast

#endif // CAST_TRANSFORM_TRANSFORM_H