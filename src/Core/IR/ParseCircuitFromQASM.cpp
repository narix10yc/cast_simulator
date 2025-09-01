#include "cast/Core/IRNode.h"
#include "cast/Transform/Transform.h"
#include "openqasm/Parser.h"

using namespace cast;

cast::MaybeError<ir::CircuitNode>
cast::parseCircuitFromQASMFile(const std::string& qasmFileName) {
  openqasm::Parser parser(qasmFileName);
  auto qasmRoot = parser.parse();
  if (qasmRoot == nullptr) {
    return makeError("Failed to parse QASM file: " +
                                            qasmFileName);
  }
  ast::ASTContext astCtx;
  auto astCircuit = transform::cvtQasmCircuitToAstCircuit(*qasmRoot, astCtx);
  if (astCircuit == nullptr) {
    return makeError(
        "Failed to convert QASM AST to CAST AST: " + qasmFileName);
  }

  auto circuit = transform::cvtAstCircuitToIrCircuit(*astCircuit, astCtx);
  if (circuit == nullptr) {
    return makeError(
        "Failed to convert CAST AST to CAST IR: " + qasmFileName);
  }
  return std::move(*circuit);
}