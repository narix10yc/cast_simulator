#include "openqasm/parser.h"

using namespace openqasm;

std::unique_ptr<ast::Expression> Parser::parseExpr() {
  logDebug(2, "Expression: ready to parse; curToken = " + curToken.toString());

  auto lhs = parsePrimaryExpr();
  if (!lhs) {
    logError("Expression: unable to parse lhs");
    return nullptr;
  }
  logDebug(3, "Expression: parsed lhs " + lhs->toString());

  BinaryOp binop = curToken.toBinaryOp();
  if (binop == BinaryOp::None) { // Not a binop
    logDebug(3, "Not a binop, return lhs");
    return lhs;
  }

  nextToken(); // eat the binop
  return parseExprRHS(binop, std::move(lhs));
}

std::unique_ptr<ast::Expression>
Parser::parseExprRHS(BinaryOp lhsBinop,
                     std::unique_ptr<ast::Expression>&& lhs) {
  logDebug(3, "Expression: ready to parse rhs");
  auto rhs = parsePrimaryExpr();
  if (!rhs) {
    logError("Expression: unable to parse rhs");
    return nullptr;
  }
  logDebug(3, "Expression: parsed rhs " + rhs->toString());

  BinaryOp binop = curToken.toBinaryOp();
  if (binop == BinaryOp::None) {
    logDebug(3, "Expression: No more binop, return");
    auto ret = std::make_unique<ast::BinaryExpr>(lhsBinop, std::move(lhs),
                                                 std::move(rhs));
    return ret;
  }

  // Encounter another binop
  nextToken(); // eat the binop
  if (getBinopPrecedence(binop) > getBinopPrecedence(lhsBinop)) {
    auto newRHS = parseExprRHS(binop, std::move(rhs));
    return std::make_unique<ast::BinaryExpr>(lhsBinop, std::move(lhs),
                                             std::move(newRHS));
  } else {
    auto newLHS = std::make_unique<ast::BinaryExpr>(lhsBinop, std::move(lhs),
                                                    std::move(rhs));
    return parseExprRHS(binop, std::move(newLHS));
  }
}

std::unique_ptr<ast::Expression> Parser::parsePrimaryExpr() {
  UnaryOp unaryOp = UnaryOp::None;
  if (curToken.type == TokenTy::Add) {
    unaryOp = UnaryOp::Positive;
    nextToken();
  } else if (curToken.type == TokenTy::Sub) {
    unaryOp = UnaryOp::Negative;
    nextToken();
  }

  std::unique_ptr<ast::Expression> expr;
  if (curToken.type == TokenTy::Numeric)
    expr = parseNumericExpr();
  else if (curToken.type == TokenTy::Identifier)
    expr = parseVariableExpr();
  else if (curToken.type == TokenTy::L_RoundBracket)
    expr = parseParenExpr();
  else {
    logError("Unknown token when expecting a primary expression");
    return nullptr;
  }

  if (unaryOp == UnaryOp::None)
    return expr;
  return std::make_unique<ast::UnaryExpr>(unaryOp, std::move(expr));
}

std::unique_ptr<ast::NumericExpr> Parser::parseNumericExpr() {
  if (curToken.type != TokenTy::Numeric) {
    logError("Expect numerics when parsing Numerics Expression");
    return nullptr;
  }
  auto expr = std::make_unique<ast::NumericExpr>(std::stod(curToken.str));
  nextToken();
  return expr;
}

std::unique_ptr<ast::Expression> Parser::parseVariableExpr() {
  auto next = peek();

  if (next.type == TokenTy::L_RoundBracket) {
    // funcCall
    logError("funcCall not implemented yet");
    return nullptr;
  } else if (next.type == TokenTy::L_SquareBracket) {
    // subscript
    return parseSubscriptExpr();
  }

  // variable expression
  auto name = curToken.str;
  nextToken();
  return std::make_unique<ast::VariableExpr>(name);
}

std::unique_ptr<ast::SubscriptExpr> Parser::parseSubscriptExpr() {
  auto name = curToken.str;
  if (!expectNextToken(TokenTy::L_SquareBracket)) {
    logError("Subscript expr: expect '['");
    return nullptr;
  }

  nextToken(); // eat '['
  auto numericExpr = parseNumericExpr();
  if (!numericExpr) {
    logError("Expect index");
    return nullptr;
  }
  int index = static_cast<int>(numericExpr->getValue());
  if (curToken.type != TokenTy::R_SquareBracket) {
    logError("Subscript Expression: expect ']'");
    return nullptr;
  }

  nextToken(); // eat ']'
  return std::make_unique<ast::SubscriptExpr>(name, index);
}

std::unique_ptr<ast::Expression> Parser::parseParenExpr() {
  nextToken(); // Eat '('
  auto expr = parseExpr();
  if (!expr) {
    return nullptr;
  }
  if (curToken.type != TokenTy::R_RoundBracket) {
    logError("Expected ')'");
    return nullptr;
  }
  nextToken(); // Eat ')'
  return expr;
}
