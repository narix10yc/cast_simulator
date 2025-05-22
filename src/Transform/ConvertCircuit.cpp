#include "cast/Transform/Transform.h"
#include "llvm/Support/Casting.h"

using namespace cast;
using namespace cast::draft;

static std::unique_ptr<ir::IfMeasureNode> convertIfMeasure(
    ast::IfStmt* astIf, ast::ASTContext& astCtx);

/// @brief Convert a span of AST statements to IR statements. Returns the number
/// of statements converted. The IR statements are appended to the provided
/// vector \c irStmts. 
static int convertSpanOfStmts(
    std::span<ast::Stmt*> astStmts, ast::ASTContext& astCtx,
    std::vector<std::unique_ptr<ir::IRNode>>& irStmts) {
  if (astStmts.empty())
    return 0;

  const auto initialSize = irStmts.size();
  std::unique_ptr<ir::CircuitGraphNode> irCircuitGraphNode =
    std::make_unique<ir::CircuitGraphNode>();
  const auto cutCircuitGraphNode = [&]() {
    assert(irCircuitGraphNode != nullptr);
    irStmts.push_back(std::move(irCircuitGraphNode));
    irCircuitGraphNode = std::make_unique<ir::CircuitGraphNode>();
  };
  const auto appendGateToCircuitGraphNode = [&](ast::GateApplyStmt* astGate) {
    assert(irCircuitGraphNode != nullptr);
  };

  for (auto it = astStmts.begin(), end = astStmts.end(); it != end; ++it) {
    // AST IfStmt acts like a barrier. It cuts current circuit graph node
    if (auto* astIf = llvm::dyn_cast<ast::IfStmt>(*it)) {
      cutCircuitGraphNode();
      auto irIf = convertIfMeasure(astIf, astCtx);
      assert(irIf);
      irStmts.push_back(std::move(irIf));
      continue;
    }
    // otherwise, append a gate to the circuit graph node
    auto* astGateChain = llvm::dyn_cast<ast::GateChainStmt>(*it);
    // astGateChain->gates

  }
}

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

std::unique_ptr<ir::CircuitNode> transform::convertCircuit(
    const ast::CircuitStmt& astCircuit, ast::ASTContext& astCtx) {
  auto* irCircuit = new ir::CircuitNode(std::string(astCircuit.name.str));


  return std::unique_ptr<ir::CircuitNode>(irCircuit);
}