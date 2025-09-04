#include "cast/Core/IRNode.h"
#include "cast/Transform/Transform.h"
#include "openqasm/Parser.h"
#include "llvm/Support/Error.h"

using namespace cast;

llvm::Expected<std::unique_ptr<ir::CircuitNode>>
cast::parseCircuitFromQASMFile(const std::string& qasmFileName) {
  openqasm::Parser parser(qasmFileName);
  auto qasmRoot = parser.parse();
  if (qasmRoot == nullptr) {
    return llvm::createStringError("Failed to parse QASM file: " +
                                   qasmFileName);
  }
  ast::ASTContext astCtx;
  auto astCircuit = transform::cvtQasmCircuitToAstCircuit(*qasmRoot, astCtx);
  if (astCircuit == nullptr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to convert QASM AST to CAST AST: " + qasmFileName);
  }

  auto circuit = transform::cvtAstCircuitToIrCircuit(*astCircuit, astCtx);
  if (circuit == nullptr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to convert CAST AST to CAST IR: " + qasmFileName);
  }
  return circuit;
}