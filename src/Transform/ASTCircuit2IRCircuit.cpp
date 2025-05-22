#include "cast/Transform/ASTCircuit2IRCircuit.h"
#include "llvm/Support/Casting.h"

using namespace cast;
using namespace cast::draft;

static std::unique_ptr<ir::IRNode> convert(
    ast::Stmt* astStmt, ast::ASTContext& astCtx);

static std::unique_ptr<ir::IfMeasureNode> convertIfMeasure(
    ast::IfStmt* astIf, ast::ASTContext& astCtx) {
  auto* astMeasureExpr = 
    llvm::dyn_cast<draft::ast::MeasureExpr>(astIf->condition);
  if (astMeasureExpr == nullptr) {
    std::cerr << "Error: If condition must be a measure expression\n";
    return nullptr;
  }
  auto* astTargetQubitExpr =
    llvm::dyn_cast<ast::IntegerLiteral>(astMeasureExpr->target);
  if (astTargetQubitExpr == nullptr) {
    std::cerr << "Error: Measure target must be an integer literal\n";
    return nullptr;
  }
  auto* irNode = new ir::IfMeasureNode(astTargetQubitExpr->value);

  return std::unique_ptr<ir::IfMeasureNode>(irNode);
}

  
static std::unique_ptr<ir::IRNode> convert(
    ast::Stmt* astStmt, ast::ASTContext& astCtx) {
  if (auto* s = llvm::dyn_cast<ast::IfStmt>(astStmt)) {
    return convertIfMeasure(s, astCtx);
  }
  assert(false && "Unsupported statement type");
  return nullptr;
}

std::unique_ptr<ir::CircuitNode> transform::ASTCircuit2IRCircuit(
    const ast::CircuitStmt& astCircuit, ast::ASTContext& astCtx) {
  auto* irCircuit = new ir::CircuitNode(std::string(astCircuit.name.str));


  return std::unique_ptr<ir::CircuitNode>(irCircuit);
}