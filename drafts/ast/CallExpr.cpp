#include "new_parser/Parser.h"
#include "utils/utils.h"

using namespace cast::draft;

std::ostream& ast::CallExpr::print(std::ostream& os) const {
  os << name << "(";
  utils::printSpanWithPrinterNoBracket(
    os, args, [](std::ostream& os, ast::Expr* arg) { arg->print(os); });
  return os << ")";
}