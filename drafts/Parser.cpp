#include "new_parser/Parser.h"
#include <fstream>
#include <cassert>

using namespace cast::draft;

void Parser::requireCurTokenIs(TokenKind kind, const char* msg) const {
  if (curToken.is(kind))
    return;
  if (msg)
    logErrHere(msg);
  else {
    std::stringstream ss;
    ss << "Requires a '" << internal::getNameOfTokenKind(kind) << "' token"
       << " (Got '" << internal::getNameOfTokenKind(curToken.kind) << "')\n";
    logErrHere(ss.str().c_str());
  }
  failAndExit();
}

std::complex<double> Parser::parseComplexNumber() {
  double multiple = 1.0;
  // general complex number, parenthesis required
  if (optionalAdvance(tk_L_RoundBracket)) {
    if (curToken.is(tk_Sub)) {
      advance(tk_Sub);
      multiple = -1.0;
    }
    requireCurTokenIs(tk_Numeric);
    double re = multiple * curToken.toDouble();
    advance(tk_Numeric);

    if (curToken.is(tk_Add))
      multiple = 1.0;
    else if (curToken.is(tk_Sub))
      multiple = -1.0;
    else
      logErrHere("Expect '+' or '-' when parsing a general complex number");
    advance();

    requireCurTokenIs(tk_Numeric);
    double im = multiple * curToken.toDouble();
    advance(tk_Numeric);

    requireCurTokenIs(tk_R_RoundBracket);
    advance(tk_R_RoundBracket);
    return {re, im};
  }

  multiple = 1.0;
  if (curToken.is(tk_Sub)) {
    advance(tk_Sub);
    multiple = -1.0;
  }

  // i or -i
  if (curToken.isI()) {
    advance(tk_Identifier);
    return {0.0, multiple};
  }

  // purely real or purely imaginary
  if (curToken.is(tk_Numeric)) {
    double value = multiple * curToken.toDouble();
    advance(tk_Numeric);
    if (curToken.isI()) {
      advance(tk_Identifier);
      return {0.0, value};
    }
    return {value, 0.0};
  }

  logErrHere("Unable to parse complex number");
  return 0.0;
}

// Try to convert a general expression to a simple numeric expression.
// Return nullptr if the conversion is not possible.
static ast::SimpleNumericExpr*
convertExprToSimpleNumeric(ASTContext& ctx, ast::Expr* expr) {
  if (expr == nullptr)
    return nullptr;
  // okay: SimpleNumericExpr
  if (auto* e = llvm::dyn_cast<ast::SimpleNumericExpr>(expr))
    return e;

  // possibly okay: MinusOpExpr
  if (auto* e = llvm::dyn_cast<ast::MinusOpExpr>(expr)) {
    auto* operand = convertExprToSimpleNumeric(ctx, e->operand);
    if (operand == nullptr)
      return nullptr;
    return ast::SimpleNumericExpr::neg(ctx, operand);
  }

  // possibly okay: BinaryOpExpr
  if (auto* e = llvm::dyn_cast<ast::BinaryOpExpr>(expr)) {
    auto* lhs = convertExprToSimpleNumeric(ctx, e->lhs);
    auto* rhs = convertExprToSimpleNumeric(ctx, e->rhs);
    if (!lhs || !rhs)
      return nullptr;
    switch (e->op) {
      case ast::BinaryOpExpr::Add:
        return ast::SimpleNumericExpr::add(ctx, lhs, rhs);
      case ast::BinaryOpExpr::Sub:
        return ast::SimpleNumericExpr::sub(ctx, lhs, rhs);
      case ast::BinaryOpExpr::Mul:
        return ast::SimpleNumericExpr::mul(ctx, lhs, rhs);
      case ast::BinaryOpExpr::Div:
        return ast::SimpleNumericExpr::div(ctx, lhs, rhs);
      default:
        assert(false && "Illegal binary operator");
        return nullptr;
    }
  }

  // otherwise, not convertible to simple numeric
  return nullptr;
}

ast::Attribute* Parser::parseAttribute() {
  if (curToken.isNot(tk_Less))
    return nullptr;
  ast::IntegerLiteral* nQubits = nullptr;
  ast::IntegerLiteral* nParams = nullptr;
  ast::SimpleNumericExpr* phase = nullptr;
  
  advance(tk_L_SquareBracket);
  while (curToken.isNot(tk_R_SquareBracket)) {
    requireCurTokenIs(tk_Identifier, "Attribute name expected");
    auto name = curToken.toString();
    advance(tk_Identifier);
    requireCurTokenIs(tk_Equal, "Expect '=' after attribute name");
    advance(tk_Equal);

    // 'nqubits', 'nparams', and 'phase' are reserved attributes
    if (name == "nqubits") {
      requireCurTokenIs(tk_Numeric, "Attribute 'nqubits' must be a number");
      nQubits = new (ctx) ast::IntegerLiteral(curToken.toInt());
      advance(tk_Numeric);
    }
    else if (name == "nparams") {
      requireCurTokenIs(tk_Numeric, "Attribute 'nparams' must be a number");
      nParams = new (ctx) ast::IntegerLiteral(curToken.toInt());
      advance(tk_Numeric);
    }
    else if (name == "phase") {
      auto* expr = parseExpr();
      phase = convertExprToSimpleNumeric(ctx, expr);
      std::cerr << "Successful: ";
      phase->print(std::cerr) << "\n";
      if (phase == nullptr) {
        logErrHere("Attribute 'phase' must be a simple numeric expression");
        failAndExit();
      }
    }
    else {
      // other attributes
      assert(false && "Not Implemented yet");
    }
    optionalAdvance(tk_Comma);
  }
  advance(tk_Greater);
  return new (ctx) ast::Attribute(nQubits, nParams, phase);
}

ast::CircuitStmt* Parser::parseCircuitStmt() {
  assert(curToken.is(tk_Circuit) &&
         "parseCircuitStmt expects to be called with a 'Circuit' token");
  advance(tk_Circuit);

  auto* attr = parseAttribute();
  requireCurTokenIs(tk_Identifier, "Expect a circuit name");
  auto name = ctx.createIdentifier(curToken.toStringView());
  advance(tk_Identifier);

  requireCurTokenIs(tk_L_CurlyBracket, "Expect '{' to start circuit body");
  advance(tk_L_CurlyBracket);
  // circuit body
  llvm::SmallVector<ast::Stmt*> body;
  while (true) {
    auto* stmt = parseCircuitLevelStmt();
    if (stmt == nullptr)
      break;
    body.push_back(stmt);
  }

  requireCurTokenIs(tk_R_CurlyBracket, "Expect '}' to end circuit body");
  advance(tk_R_CurlyBracket);

  return new (ctx) ast::CircuitStmt(
    name, 
    attr,
    ctx.createSpan(body.data(), body.size())
  );
}

ast::Stmt* Parser::parseCircuitLevelStmt() {
  switch (curToken.kind) {
    case tk_Measure: {
      advance(tk_Measure);
      auto* target = parseExpr();
      assert(target != nullptr);
      return new (ctx) ast::MeasureStmt(target);
    }
    case tk_Identifier:
      return parseGateChainStmt();
    default:
      return nullptr;
  }
}

ast::GateChainStmt* Parser::parseGateChainStmt() {
  llvm::SmallVector<ast::GateApplyStmt*> gates;
  while (true) {
    if (curToken.is(tk_Semicolon))
      break;
    auto* gate = parseGateApplyStmt();
    assert(gate != nullptr);
    gates.push_back(gate);
    if (curToken.is(tk_AtSymbol)) {
      advance(tk_AtSymbol);
      continue;
    }
    requireCurTokenIs(tk_Semicolon, "Expect ';' to finish a GateChain");
  }
  advance(tk_Semicolon);

  return new (ctx) ast::GateChainStmt(
    ctx.createSpan(gates.data(), gates.size())
  );
}

ast::GateApplyStmt* Parser::parseGateApplyStmt() {
  // name
  assert(curToken.is(tk_Identifier) &&
         "parseGateApplyStmt expects to be called with an identifier");
  auto name = ctx.createIdentifier(curToken.toStringView());
  advance(tk_Identifier);
  
  llvm::SmallVector<ast::Expr*> params;
  llvm::SmallVector<ast::Expr*> qubits;
  // gate parameters
  if (curToken.is(tk_L_RoundBracket)) {
    advance(tk_L_RoundBracket);
    while (true) {
      auto* expr = parseExpr();
      if (expr == nullptr)
        break;
      // try to simplify the gate parameter to simple numerics
      if (auto* e = convertExprToSimpleNumeric(ctx, expr))
        expr = e;
      params.push_back(expr);
      if (curToken.is(tk_R_RoundBracket))
        break;
      requireCurTokenIs(tk_Comma, "Expect ',' to separate parameters");
      advance(tk_Comma);
      continue;
    }
    advance(tk_R_RoundBracket);
  }
  
  // target qubits
  while (true) {
    auto* expr = parseExpr();
    if (expr == nullptr)
      break;
    qubits.push_back(expr);
    // skip optional comma
    if (curToken.is(tk_Comma)) {
      advance(tk_Comma);
      continue;
    }
  }

  return new (ctx) ast::GateApplyStmt(
    name, 
    ctx.createSpan(params.data(), params.size()),
    ctx.createSpan(qubits.data(), qubits.size())
  );
}

ast::RootNode* Parser::parse() {
  llvm::SmallVector<ast::Stmt*> stmts;

  while (curToken.isNot(tk_Eof)) {
    switch (curToken.kind) {
      case tk_Circuit: {
        auto* stmt = parseCircuitStmt();
        assert(stmt != nullptr);
        stmts.push_back(stmt);
        break;
      }
      default: {
        logErrHere("Expecting a top-level expression, which could be a "
                   "Circuit or Channel.");
        failAndExit();
      }
    }
  }
  return new (ctx) ast::RootNode(
    ctx.createSpan(stmts.data(), stmts.size())
  );
}