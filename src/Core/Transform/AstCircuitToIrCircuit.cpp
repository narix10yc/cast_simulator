#include "cast/Transform/Transform.h"
#include "utils/iocolor.h"
#include "llvm/Support/Casting.h"

using namespace cast;
using namespace cast;

// forward declaration
static std::unique_ptr<ir::IfMeasureNode>
convertIfMeasure(ast::IfStmt* astIf, ast::ASTContext& astCtx);

/// @brief Convert a span of AST statements to IR statements. Returns the number
/// of statements converted. The IR statements are appended to the provided
/// vector \c irStmts.
static int convertSpanOfStmts(std::span<ast::Stmt*> astStmts,
                              ast::ASTContext& astCtx,
                              ir::CompoundNode& irCompoundNode) {
  if (astStmts.empty())
    return 0;

  const auto initialSize = irCompoundNode.size();
  std::unique_ptr<ir::CircuitGraphNode> irCircuitGraphNode =
      std::make_unique<ir::CircuitGraphNode>();

  const auto cutCircuitGraphNode = [&]() {
    assert(irCircuitGraphNode != nullptr);
    irCompoundNode.push_back(std::move(irCircuitGraphNode));
    irCircuitGraphNode = std::make_unique<ir::CircuitGraphNode>();
  };

  const auto appendGateToCircuitGraphNode = [&](ast::GateApplyStmt* astGate) {
    assert(irCircuitGraphNode != nullptr);
    auto gateMatrix = transform::cvtAstGateToGateMatrix(astGate, astCtx);
    assert(gateMatrix != nullptr && "Failed to convert astGate to gateMatrix");
    QuantumGate::TargetQubitsType qubits;
    for (auto* qubitExpr : astGate->qubits) {
      auto* qubitLit = llvm::dyn_cast<ast::IntegerLiteral>(qubitExpr);
      if (qubitLit == nullptr) {
        std::cerr << "Error: Qubit index must be an integer literal\n";
        return;
      }
      qubits.push_back(qubitLit->value);
    }
    auto qGate = StandardQuantumGate::Create(gateMatrix, nullptr, qubits);
    irCircuitGraphNode->insertGate(qGate);
  };

  // main loop
  for (auto it = astStmts.begin(), end = astStmts.end(); it != end; ++it) {
    // AST IfStmt acts like a barrier. It cuts current circuit graph node
    if (auto* astIf = llvm::dyn_cast<ast::IfStmt>(*it)) {
      cutCircuitGraphNode();
      auto irIf = convertIfMeasure(astIf, astCtx);
      assert(irIf);
      irCompoundNode.push_back(std::move(irIf));
      continue;
    }
    // otherwise, append gate(s) to the circuit graph node
    if (auto* astGateChain = llvm::dyn_cast<ast::GateChainStmt>(*it)) {
      for (auto* astGate : astGateChain->gates)
        appendGateToCircuitGraphNode(astGate);
      continue;
    }
    if (auto* astGate = llvm::dyn_cast<ast::GateApplyStmt>(*it)) {
      appendGateToCircuitGraphNode(astGate);
      continue;
    }
    std::cerr << YELLOW("Warning: ") << "In converting AST to IR, "
              << "skipped unsupported statement type: " << (*it)->getKindName()
              << "\n";
  }
  if (irCircuitGraphNode->nGates() > 0)
    cutCircuitGraphNode();
  return irCompoundNode.size() - initialSize;
}

static std::unique_ptr<ir::IfMeasureNode>
convertIfMeasure(ast::IfStmt* astIf, ast::ASTContext& astCtx) {
  auto* astMeasureExpr = llvm::dyn_cast<ast::MeasureExpr>(astIf->condition);
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
  convertSpanOfStmts(astIf->thenBody, astCtx, irNode->thenBody);
  convertSpanOfStmts(astIf->elseBody, astCtx, irNode->elseBody);

  return std::unique_ptr<ir::IfMeasureNode>(irNode);
}

std::unique_ptr<ir::CircuitNode>
transform::cvtAstCircuitToIrCircuit(const ast::CircuitStmt& astCircuit,
                                    ast::ASTContext& astCtx) {
  auto* irCircuit = new ir::CircuitNode(std::string(astCircuit.name.str));
  auto nCvted = convertSpanOfStmts(astCircuit.body, astCtx, irCircuit->body);

  return std::unique_ptr<ir::CircuitNode>(irCircuit);
}