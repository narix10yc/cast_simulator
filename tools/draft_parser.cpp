#include "cast/Core/AST/Parser.h"
#include "cast/Transform/Transform.h"
#include "cast/Core/Optimize.h"

using namespace cast;

static const char* Program = R"(
Circuit my_circuit {
  H 0;
  CX 0 1;
  CX 0 2;
  RZ(Pi/2) 1;
  // RZ(Pi/3) 2;
  If (Measure 0) {
    X 0;
    RZ(Pi/4) 1;
    RZ(Pi/5) 2;
    H 0;
    CX 0 1;
  }
  Else {
    H 0;
    RZ(Pi/6) 1;
    RZ(Pi/7) 2;
    CX 0 2;
  }
  RZ(Pi/8) 1;
  RZ(Pi/9) 2;
  CX 1 0;
  CX 2 0;
  Out (Measure 0);
}
)";

int main(int argc, char** argv) {
  ast::ASTContext astCtx;
  astCtx.loadRawBuffer(Program);
  // astCtx.displayLineTable(std::cerr);
  ast::Parser parser(astCtx);
  
  // parser.loadFromFile(argv[1]);

  auto* root = parser.parse();
  ast::PrettyPrinter p(std::cerr);
  // root->prettyPrint(p, 0);
  // root->print(std::cerr);
  const auto* astCircuit = root->lookupCircuit("my_circuit");
  assert(astCircuit != nullptr && "Failed to find circuit my_circuit");

  auto irCircuit = cast::transform::cvtAstCircuitToIrCircuit(*astCircuit, astCtx);

  if (irCircuit == nullptr) {
    std::cerr << "Failed to transform AST to IR\n";
    return 1;
  }
  irCircuit->visualize(std::cerr);

  cast::applyCanonicalizationPass(*irCircuit, 1e-8);
  irCircuit->displayInfo(std::cerr << "\nAfter Canonicalization\n", 1);
  irCircuit->visualize(std::cerr);

  return 0;
}