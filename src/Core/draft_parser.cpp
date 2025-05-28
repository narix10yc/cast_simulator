#include "cast/Core/AST/Parser.h"
#include "cast/Transform/Transform.h"
#include "cast/Fusion.h"

using namespace cast::draft;
using namespace cast;

static const char* Program = R"(
Circuit my_circuit {
  H 0;
  CX 0 1;
  If (Measure 0) {
    X 0;
  }
  Else {
    X 1;
  }
  RZ(Pi/4) 0;
  Out (Measure 0);
}
)";

int main(int argc, char** argv) {
  ast::ASTContext astCtx;
  astCtx.loadRawBuffer(Program);
  astCtx.displayLineTable(std::cerr);
  ast::Parser parser(astCtx);
  
  // parser.loadFromFile(argv[1]);

  auto* root = parser.parse();
  ast::PrettyPrinter p(std::cerr);
  root->prettyPrint(p, 0);
  root->print(std::cerr);
  const auto* astCircuit = root->lookupCircuit("my_circuit");
  assert(astCircuit != nullptr && "Failed to find circuit my_circuit");

  auto irCircuit = cast::transform::cvtAstCircuitToIrCircuit(*astCircuit, astCtx);

  if (irCircuit == nullptr) {
    std::cerr << "Failed to transform AST to IR\n";
    return 1;
  }
  irCircuit->print(std::cerr, 0);
  irCircuit->displayInfo(std::cerr, 3);

  return 0;
}