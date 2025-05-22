#include "new_parser/Parser.h"
#include "cast/Transform/ASTCircuit2IRCircuit.h"

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
  ast::ASTContext context;
  ast::Parser parser(context);
  parser.loadRawBuffer(Program);
  parser.displayLineTable();

  // parser.loadFromFile(argv[1]);

  auto* root = parser.parse();
  ast::PrettyPrinter p(std::cerr);
  root->prettyPrint(p, 0);
  root->print(std::cerr);
  const auto* astCircuit = root->lookupCircuit("my_circuit");
  assert(astCircuit != nullptr && "Failed to find circuit my_circuit");

  auto irCircuit = cast::transform::ASTCircuit2IRCircuit(*astCircuit, context);

  if (irCircuit == nullptr) {
    std::cerr << "Failed to transform AST to IR\n";
    return 1;
  }
  irCircuit->print(std::cerr);

  return 0;
}