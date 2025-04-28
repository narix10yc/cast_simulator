#include "new_parser/Parser.h"

using namespace cast::draft;

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

static int getBinaryOpPrecedence(ast::BinaryOpExpr::BinaryOpKind binOp) {
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

static constexpr int minusOpPrecedence = 30;

static ast::BinaryOpExpr::BinaryOpKind toBinaryOp(TokenKind tokenKind) {
  switch (tokenKind) {
  case tk_Add:
    return ast::BinaryOpExpr::Add;
  case tk_Sub:
    return ast::BinaryOpExpr::Sub;
  case tk_Mul:
    return ast::BinaryOpExpr::Mul;
  case tk_Div:
    return ast::BinaryOpExpr::Div;
  case tk_Pow:
    return ast::BinaryOpExpr::Pow;
  default:
    return ast::BinaryOpExpr::Invalid;
  }
}

ast::Expr* Parser::parsePrimaryExpr() {
  // handle unary operators (+ and -)
  switch (curToken.kind) {
  case tk_Add: {
    advance(tk_Add);
    return parsePrimaryExpr();
  }
  case tk_Sub: {
    advance(tk_Sub);
    return new (ctx) ast::MinusOpExpr(parseExpr(minusOpPrecedence));
  }
  // Identifier
  case tk_Identifier: {
    auto name = ctx.createIdentifier(curToken.toStringView());
    advance(tk_Identifier);
    return new (ctx) ast::IdentifierExpr(name);
  }
  // Numerics
  case tk_Numeric: {
    if (curToken.convertibleToInt()) {
      auto iValue = curToken.toInt();
      advance(tk_Numeric);
      return new (ctx) ast::IntegerLiteral(iValue);
    }
    auto fValue = curToken.toDouble();
    advance(tk_Numeric);
    return new (ctx) ast::FloatingLiteral(fValue);
  }
  // Pi
  case tk_Pi: {
    advance(tk_Pi);
    return new (ctx) ast::FractionPiLiteral(1, 1);
  }
  // Measure
  case tk_Measure: {
    advance(tk_Measure);
    return new (ctx) ast::MeasureExpr(parseExpr());
  }
  // All
  case tk_All: {
    advance(tk_All);
    return new (ctx) ast::AllExpr();
  }
  // Parameter (#number)
  case tk_Hash: {
    advance(tk_Hash);
    requireCurTokenIs(tk_Numeric, "Expect a number after #");
    auto index = curToken.toInt();
    advance(tk_Numeric);
    return new (ctx) ast::ParameterExpr(index);
  }
  // Paranthesis 
  case tk_L_RoundBracket: {
    advance(tk_L_RoundBracket);
    auto* expr = parseExpr();
    requireCurTokenIs(tk_R_RoundBracket, "Expect ')' to close the expression");
    advance(tk_R_RoundBracket);
    return expr;
  }
  default:
    return nullptr;
  }
}

ast::Expr* Parser::parseExpr(int precedence) {
  auto* lhs = parsePrimaryExpr();
  if (lhs == nullptr)
    return nullptr;
  while (true) {
    auto binOp = toBinaryOp(curToken.kind);
    int prec = getBinaryOpPrecedence(binOp);
    if (prec < precedence)
      break;
    advance();
    auto* rhs = parseExpr(prec + 1);
    if (rhs == nullptr) {
      logErrHere("Missing RHS of a binary expression");
      failAndExit();
    }
    lhs = new (ctx) ast::BinaryOpExpr(binOp, lhs, rhs);
  }
  return lhs;
}