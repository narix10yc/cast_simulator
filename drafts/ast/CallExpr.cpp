#include "new_parser/Parser.h"
#include "utils/utils.h"

using namespace cast::draft;

std::ostream& ast::CallExpr::print(std::ostream& os) const {
  os << name << "(";
  utils::printSpanWithPrinterNoBracket(
    os, args, [](std::ostream& os, ast::Expr* arg) { arg->print(os); });
  return os << ")";
}

void ast::CallExpr::prettyPrint(PrettyPrinter& p, int indent) const {
  p.write(indent)
    << "CallExpr{name=" << this->name
    << ", " << args.size() << " args}\n";
              
  p.setState(indent, args.size());
  for (const auto* arg : args)
    arg->prettyPrint(p, indent + 1);
}