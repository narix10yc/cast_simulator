#include "openqasm/parser.h"
#include "cast/Transform/Transform.h"

using namespace cast;
using namespace cast::draft;

int main(int argc, char** argv) {
  assert(argc > 1 && "Usage: cpu_bcmk <qasm_file>");
  std::string qasmFile = argv[1];
  openqasm::Parser qasmParser(qasmFile, /* debugLevel */ 0);
  auto qasmRoot = qasmParser.parse();

  ast::ASTContext astCtx;
  auto castCircuit = transform::cvtQasmCircuitToAstCircuit(*qasmRoot, astCtx);

  castCircuit->updateAttribute();
  auto irCircuit = transform::cvtAstCircuitToIrCircuit(*castCircuit, astCtx);

  irCircuit->print(std::cerr, 0);
  return 0;
}