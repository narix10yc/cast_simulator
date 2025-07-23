#include "cast/Core/IRNode.h"
#include "cast/Transform/Transform.h"
#include "openqasm/parser.h"

using namespace cast;

cast::MaybeError<ir::CircuitNode>
cast::parseCircuitFromQASMFile(const std::string& qasmFileName) {
  openqasm::Parser parser(qasmFileName);
  auto qasmRoot = parser.parse();
  if (qasmRoot == nullptr) {
    return cast::makeError<ir::CircuitNode>("Failed to parse QASM file: " +
                                            qasmFileName);
  }
  ast::ASTContext astCtx;
  auto astCircuit = transform::cvtQasmCircuitToAstCircuit(*qasmRoot, astCtx);
  if (astCircuit == nullptr) {
    return cast::makeError<ir::CircuitNode>(
        "Failed to convert QASM AST to CAST AST: " + qasmFileName);
  }

  auto circuit = transform::cvtAstCircuitToIrCircuit(*astCircuit, astCtx);
  if (circuit == nullptr) {
    return cast::makeError<ir::CircuitNode>(
        "Failed to convert CAST AST to CAST IR: " + qasmFileName);
  }
  return std::move(*circuit);
}