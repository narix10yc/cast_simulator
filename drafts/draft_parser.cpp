#include "new_parser/Parser.h"

using namespace cast::draft;

int main(int argc, char** argv) {
  assert(argc > 1);
  ASTContext context;
  Parser parser(context, argv[1]);
  auto* root = parser.parse();
  root->print(std::cerr);

  return 0;
}