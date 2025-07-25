#include "cast/Core/AST/Parser.h"

using namespace cast::ast;

void FractionLiteral::prettyPrint(PrettyPrinter& p, int indent) const {
  p.write(indent) << getKindName() << "(";
  print(p.os) << ")\n";
}