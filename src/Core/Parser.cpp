#include "llvm/Support/Casting.h"
#include "cast/Core/AST/Parser.h"
#include <fstream>
#include <cassert>

using namespace cast::ast;

void Parser::addSymbol(Identifier name, Node* node) {
  const auto locPrint = [this](Node* node) {
    if (auto* circuit = llvm::dyn_cast<CircuitStmt>(node)) {
      ctx.printLineInfo(std::cerr, circuit->nameLoc);
    } else if (auto* channel = llvm::dyn_cast<ChannelStmt>(node)) {
      ctx.printLineInfo(std::cerr, channel->nameLoc);
    } else {
      assert(false && "Unknown node type");
    }
  };
  assert(currentScope && "Trying to add symbol to a null scope");
  auto it = currentScope->symbols.find(name);
  if (it != currentScope->symbols.end()) {
    err() << "Duplicate symbol: " << name << "\n";
    locPrint(node);
    std::cerr << "Previous definition:\n";
    locPrint(it->second);
    failAndExit();
  }
  currentScope->symbols[name] = node;
}

Node* Parser::lookup(Identifier name) {
  Scope* scope = currentScope;
  while (scope) {
    auto it = scope->symbols.find(name);
    if (it != scope->symbols.end())
      return it->second;
    scope = scope->parent;
  }
  return nullptr;
}

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
SimpleNumericExpr*
Parser::convertExprToSimpleNumeric(Expr* expr) {
  if (expr == nullptr)
    return nullptr;
  // okay: SimpleNumericExpr
  if (auto* e = llvm::dyn_cast<SimpleNumericExpr>(expr))
    return e;

  // possibly okay: MinusOpExpr
  if (auto* e = llvm::dyn_cast<MinusOpExpr>(expr)) {
    auto* operand = convertExprToSimpleNumeric(e->operand);
    if (operand == nullptr)
      return nullptr;
    return SimpleNumericExpr::neg(ctx, operand);
  }

  // possibly okay: BinaryOpExpr
  if (auto* e = llvm::dyn_cast<BinaryOpExpr>(expr)) {
    auto* lhs = convertExprToSimpleNumeric(e->lhs);
    auto* rhs = convertExprToSimpleNumeric(e->rhs);
    if (!lhs || !rhs)
      return nullptr;
    switch (e->op) {
      case BinaryOpExpr::Add:
        return SimpleNumericExpr::add(ctx, lhs, rhs);
      case BinaryOpExpr::Sub:
        return SimpleNumericExpr::sub(ctx, lhs, rhs);
      case BinaryOpExpr::Mul:
        return SimpleNumericExpr::mul(ctx, lhs, rhs);
      case BinaryOpExpr::Div:
        return SimpleNumericExpr::div(ctx, lhs, rhs);
      default:
        assert(false && "Illegal binary operator");
        return nullptr;
    }
  }

  // otherwise, not convertible to simple numeric
  return nullptr;
}

Attribute* Parser::parseAttribute() {
  if (curToken.isNot(tk_Less))
    return nullptr;
  IntegerLiteral* nQubits = nullptr;
  IntegerLiteral* nParams = nullptr;
  SimpleNumericExpr* phase = nullptr;
  
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
      nQubits = new (ctx) IntegerLiteral(curToken.toInt());
      advance(tk_Numeric);
    }
    else if (name == "nparams") {
      requireCurTokenIs(tk_Numeric, "Attribute 'nparams' must be a number");
      nParams = new (ctx) IntegerLiteral(curToken.toInt());
      advance(tk_Numeric);
    }
    else if (name == "phase") {
      auto* expr = parseExpr();
      phase = convertExprToSimpleNumeric(expr);
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
  return new (ctx) Attribute(nQubits, nParams, phase);
}

ParameterDeclExpr* Parser::parseParameterDecl() {
  if (curToken.isNot(tk_L_RoundBracket))
    return nullptr;
  advance(tk_L_RoundBracket);
  std::vector<IdentifierExpr*> params;
  while (true) {
    switch (curToken.kind) {
      case tk_R_RoundBracket: {
        break;
      }
      case tk_Identifier: {
        auto name = ctx.createIdentifier(curToken.toStringView());
        params.push_back(new (ctx) IdentifierExpr(name));
        advance(tk_Identifier);
        if (curToken.is(tk_Comma))
          advance(tk_Comma);
        break;
      }
      case tk_Comma: {
        logErrHere("Extra comma in parameter list");
        failAndExit();
        return nullptr;
      }
      default:
        logErrHere("Expect a parameter name or ')' to end parameter list");
        failAndExit();
    }
    if (curToken.is(tk_R_RoundBracket))
      break;
  }
  advance(tk_R_RoundBracket);
  return new (ctx) ParameterDeclExpr(
    ctx.createSpan(params.data(), params.size())
  );
}

Stmt* Parser::parseCircuitLevelStmt() {
  switch (curToken.kind) {
    case tk_Measure: {
      advance(tk_Measure);
      auto* target = parseExpr();
      assert(target != nullptr);
      return new (ctx) MeasureStmt(target);
    }
    case tk_Identifier:
      return parseGateChainStmt();
    case tk_If:
      return parseIfStmt();
    case tk_Out:
      return parseOutStmt();
    default:
      return nullptr;
  }
}

std::span<Stmt*> Parser::parseCircuitLevelStmtList() {
  std::vector<Stmt*> stmts;
  if (curToken.is(tk_L_CurlyBracket)) {
    advance(tk_L_CurlyBracket);
    while (true) {
      if (curToken.is(tk_R_CurlyBracket))
        break;
      auto* s = parseCircuitLevelStmt();
      if (s == nullptr)
        break;
      stmts.push_back(s);
    }
    advance(tk_R_CurlyBracket);
  }
  else {
    auto* s = parseCircuitLevelStmt();
    if (s == nullptr) {
      logErrHere("Expect a statement");
      failAndExit();
    }
    stmts.push_back(s);
  }

  return ctx.createSpan(stmts.data(), stmts.size());
}