#include "new_parser/Parser.h"

using namespace cast::draft;

ast::Expr* Parser::parseIdentifierOrCallExpr() {
  assert(curToken.is(tk_Identifier));
  auto name = ctx.createIdentifier(curToken.toStringView());
  advance(tk_Identifier);
  if (curToken.isNot(tk_L_RoundBracket))
    return new (ctx) ast::IdentifierExpr(name);

  // A call expr
  advance(tk_L_RoundBracket);
  std::vector<ast::Expr*> args;
  while (true) {
    auto* arg = parseExpr();
    if (arg == nullptr)
      break;
    args.push_back(arg);
    if (curToken.is(tk_Comma)) {
      advance(tk_Comma);
      continue;
    }
    if (curToken.is(tk_R_RoundBracket))
      break;
    logErrHere("Invalid token inside function call arguments");
    failAndExit();
  }

  advance(tk_R_RoundBracket);
  return new (ctx) ast::CallExpr(name, ctx.createSpan(args));
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
    auto* operand = parseExpr(ast::MinusOpExpr::getPrecedence());
    return new (ctx) ast::MinusOpExpr(operand);
  }
  // Identifier
  case tk_Identifier: {
    return parseIdentifierOrCallExpr();
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

static ast::BinaryOpExpr::BinaryOpKind toBinaryOp(TokenKind kind) {
  switch (kind) {
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

ast::Expr* Parser::parseExpr(int precedence) {
  auto* lhs = parsePrimaryExpr();
  if (lhs == nullptr)
    return nullptr;
  while (true) {
    auto binOp = toBinaryOp(curToken.kind);
    int prec = ast::BinaryOpExpr::getPrecedence(binOp);
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