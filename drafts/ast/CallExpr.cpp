#include "new_parser/Parser.h"
#include "utils/PrintSpan.h"

using namespace cast::draft;

std::ostream& ast::CallExpr::print(std::ostream& os) const {
  os << name << "(";
  utils::printSpanWithPrinterNoBracket(
    os, args, [](std::ostream& os, ast::Expr* arg) { arg->print(os); });
  return os << ")";
}

void ast::CallExpr::prettyPrint(PrettyPrinter& p, int indent) const {
  p.write(indent) << getKindName() << ": " << name << ", "
                  << args.size() << " args\n";
  p.setState(indent, args.size());
  for (const auto* arg : args)
    arg->prettyPrint(p, indent + 1);
}