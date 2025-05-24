#ifndef CAST_TRANSFORM_TRANSFORM_H
#define CAST_TRANSFORM_TRANSFORM_H

#include "cast/IR/IRNode.h"
#include "new_parser/ASTContext.h"
#include "new_parser/AST.h"
#include "openqasm/ast.h"
#include <memory>

namespace cast {

namespace transform {

draft::ast::CircuitStmt* cvtQasmCircuitToAstCircuit(
    const openqasm::ast::RootNode& qasmRoot, draft::ast::ASTContext& astCtx);

/// @brief Convert a GateApplyStmt to a GateMatrixPtr.
GateMatrixPtr cvtAstGateToGateMatrix(
    cast::draft::ast::GateApplyStmt* astGate,
    cast::draft::ast::ASTContext& astCtx);

std::unique_ptr<ir::CircuitNode> cvtAstCircuitToIrCircuit(
    const draft::ast::CircuitStmt& astCircuit, draft::ast::ASTContext& astCtx);

} // namespace transform

} // namespace cast;

#endif // CAST_TRANSFORM_TRANSFORM_H