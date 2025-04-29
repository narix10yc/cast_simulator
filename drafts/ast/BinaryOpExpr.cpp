#include "new_parser/Parser.h"
#include "new_parser/Lexer.h"

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

int ast::MinusOpExpr::getPrecedence() {
  return 30;
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

std::ostream& ast::MinusOpExpr::print(std::ostream& os) const {
  os << "-";
  operand->print(os);
  return os;
}
