#include "cast/Transform/Transform.h"
#include "llvm/Support/Casting.h"

using namespace cast;

static void capitalize(std::string& str) {
  if (str.empty())
    return;
  for (auto& c : str)
    c = std::toupper(c);
}

ast::CircuitStmt*
transform::cvtQasmCircuitToAstCircuit(const openqasm::ast::RootNode& qasmRoot,
                                      ast::ASTContext& astCtx) {
  std::vector<ast::Stmt*> stmts;

  for (const auto& s : qasmRoot.stmts) {
    if (auto* gateApplyStmt =
            llvm::dyn_cast<openqasm::ast::GateApplyStmt>(s.get())) {
      std::vector<ast::Expr*> parameters;
      std::vector<ast::Expr*> qubits;
      // parameters
      for (const auto& p : gateApplyStmt->parameters) {
        auto ev = p->getExprValue();
        assert(ev.isConstant);
        parameters.push_back(new (astCtx) ast::FloatingLiteral(ev.value));
      }
      // target qubits
      for (const auto& t : gateApplyStmt->targets) {
        qubits.push_back(new (astCtx) ast::IntegerLiteral(t->getIndex()));
      }
      auto gateName = gateApplyStmt->name;
      capitalize(gateName);
      stmts.push_back(new (astCtx)
                          ast::GateApplyStmt(astCtx.createIdentifier(gateName),
                                             astCtx.createSpan(parameters),
                                             astCtx.createSpan(qubits)));
    }
  }

  return new (astCtx)
      ast::CircuitStmt(astCtx.createIdentifier("circuit_from_qasm"),
                       LocationSpan(nullptr, nullptr), // No location info
                       nullptr,                 // No parameter declaration
                       ast::CircuitAttribute(), // No attributes
                       astCtx.createSpan(stmts));
} // transform::cvtQasmCircuitToAstCircuit