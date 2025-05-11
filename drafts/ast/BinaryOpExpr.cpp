#include "new_parser/Parser.h"

using namespace cast::draft;

int ast::BinaryOpExpr::getPrecedence(ast::BinaryOpExpr::BinaryOpKind binOp) {
  switch (binOp) {
  case ast::BinaryOpExpr::Invalid:
    return -1;
  case ast::BinaryOpExpr::Add:
  case ast::BinaryOpExpr::Sub:
    return 10;
  case ast::BinaryOpExpr::Mul:
  case ast::BinaryOpExpr::Div:
    return 20;
  case ast::BinaryOpExpr::Pow:
    return 50;
  default:
    assert(false && "Invalid binary operator");
    return -1;
  }
}

std::ostream& ast::BinaryOpExpr::print(std::ostream& os) const {
  os << "(";
  lhs->print(os);
  os << " ";
  switch (op) {
    case Add: os << "+"; break;
    case Sub: os << "-"; break;
    case Mul: os << "*"; break;
    case Div: os << "/"; break;
    case Pow: os << "**"; break;
    default: 
      assert(false && "Invalid operator");
  }
  rhs->print(os << " ");
  return os << ")";
}

void ast::BinaryOpExpr::prettyPrint(PrettyPrinter& p, int indent) const {
  p.write(indent) << getKindName() << ": ";
  switch (op) {
    case Add: p.os << "+"; break;
    case Sub: p.os << "-"; break;
    case Mul: p.os << "*"; break;
    case Div: p.os << "/"; break;
    case Pow: p.os << "**"; break;
    default:
      assert(false && "Invalid operator");
  }
  p.setState(indent, 2);
  p.os << "\n";
  p.setPrefix("lhs: ");
  lhs->prettyPrint(p, indent + 1);
  p.setPrefix("rhs: ");
  rhs->prettyPrint(p, indent + 1);
}