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

  assert(irCircuit != nullptr);
  irCircuit->displayInfo(std::cerr, 3);
  irCircuit->print(std::cerr, 0);
  
  auto circuitGraphs = irCircuit->getAllCircuitGraphs();
  for (const auto* graph : circuitGraphs) {
    graph->displayInfo(std::cerr, 3);
    // graph->visualize(std::cerr, 3);
    assert(graph->checkConsistency());
  }

  return 0;
}