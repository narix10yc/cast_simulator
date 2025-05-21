#include "new_parser/Parser.h"

using namespace cast::draft;

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
  ASTContext context;
  Parser parser(context);
  parser.loadRawBuffer(Program);

  // parser.loadFromFile(argv[1]);

  auto* root = parser.parse();
  ast::PrettyPrinter p(std::cerr);
  root->prettyPrint(p, 0);
  root->print(std::cerr);

  return 0;
}