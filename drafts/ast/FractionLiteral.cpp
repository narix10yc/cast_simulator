#include "new_parser/Parser.h"

using namespace cast::draft;

void ast::FractionLiteral::prettyPrint(PrettyPrinter& p, int indent) const {
  p.write(indent) << getKindName() << "(";
  print(p.os) << ")\n";
}