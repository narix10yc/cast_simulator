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
// This function is mainly used in parsing the 'phase' attribute of statements.
static std::unique_ptr<ast::SimpleNumericExpr>
convertExprToSimpleNumeric(const ast::Expr* expr) {
  if (expr == nullptr)
    return nullptr;
  if (auto* e = llvm::dyn_cast<ast::ParameterExpr>(expr))
    return nullptr;

  if (auto* e = llvm::dyn_cast<ast::SimpleNumericExpr>(expr))
    return std::make_unique<ast::SimpleNumericExpr>(*e);

  if (auto* e = llvm::dyn_cast<ast::MinusOpExpr>(expr)) {
    auto operand = convertExprToSimpleNumeric(e->operand.get());
    if (!operand)
      return nullptr;
    return std::make_unique<ast::SimpleNumericExpr>(-(*operand));
  }

  if (auto* e = llvm::dyn_cast<ast::BinaryOpExpr>(expr)) {
    auto lhs = convertExprToSimpleNumeric(e->lhs.get());
    auto rhs = convertExprToSimpleNumeric(e->rhs.get());
    if (!lhs || !rhs)
      return nullptr;
    switch (e->op) {
      case ast::BinaryOpExpr::Add:
        return std::make_unique<ast::SimpleNumericExpr>(*lhs + *rhs);
      case ast::BinaryOpExpr::Sub:
        return std::make_unique<ast::SimpleNumericExpr>(*lhs - *rhs);
      case ast::BinaryOpExpr::Mul:
        return std::make_unique<ast::SimpleNumericExpr>(*lhs * *rhs);
      case ast::BinaryOpExpr::Div:
        return std::make_unique<ast::SimpleNumericExpr>(*lhs / *rhs);
      default:
        assert(false && "Illegal binary operator");
        return nullptr;
    }
  }

  // otherwise, not convertible to simple numeric
  return nullptr;
}

std::unique_ptr<ast::Attribute> Parser::parseAttribute() {
  if (curToken.isNot(tk_L_SquareBracket))
    return nullptr;
  
  auto attr = std::make_unique<ast::Attribute>();

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
      attr->nQubits = curToken.toInt();
      advance(tk_Numeric);
    }
    else if (name == "nparams") {
      requireCurTokenIs(tk_Numeric, "Attribute 'nparams' must be a number");
      attr->nParams = curToken.toInt();
      advance(tk_Numeric);
    }
    else if (name == "phase") {
      auto expr = parseExpr();
      if (expr == nullptr) {
        logErrHere("Failed to parse expression for attribute 'phase'");
        failAndExit();
      }
      auto simpleNumeric = convertExprToSimpleNumeric(expr.get());
      if (simpleNumeric == nullptr) {
        std::stringstream ss;
        expr->print(ss << "Expression ")
          << " cannot be used in attribute 'phase'";
        logErrHere(ss.str().c_str());
        failAndExit();
      }
      attr->phase = *simpleNumeric;
    }
    else {
      // other attributes
      assert(false && "Not Implemented yet");
    }
    optionalAdvance(tk_Comma);
  }
  advance(tk_R_SquareBracket);
  return attr;
}

std::unique_ptr<ast::CircuitStmt> Parser::parseCircuitStmt() {
  advance(tk_Circuit);

  auto circuitStmt = std::make_unique<ast::CircuitStmt>();
  circuitStmt->attribute = std::move(parseAttribute());
  requireCurTokenIs(tk_Identifier);
  circuitStmt->name = curToken.toString();
  advance(tk_Identifier);

  advance(tk_L_CurlyBracket);
  // circuit body
  while (auto s = parseCircuitLevelStmt())
    circuitStmt->body.emplace_back(std::move(s));

  advance(tk_R_CurlyBracket);

  return circuitStmt;
}

std::unique_ptr<ast::Stmt> Parser::parseCircuitLevelStmt() {
  if (curToken.is(tk_Measure)) {
    advance(tk_Measure);
    if (curToken.isNot(tk_Numeric) || !curToken.convertibleToInt()) {
      logErrHere("Expect a target qubit after 'Measure'");
      failAndExit();
    }
    auto qubit = curToken.toInt();
    advance(tk_Numeric);
    requireCurTokenIs(tk_Semicolon);
    advance(tk_Semicolon);
    return std::make_unique<ast::MeasureStmt>(qubit);
  }
  if (curToken.is(tk_Identifier))
    return parseGateChainStmt();
  return nullptr;
}

std::unique_ptr<ast::GateChainStmt> Parser::parseGateChainStmt() {
  auto chain = std::make_unique<ast::GateChainStmt>();
  const auto appendGateApplyStmt = [&]() {
    // name
    assert(curToken.is(tk_Identifier));
    chain->gates.emplace_back(curToken.toString());
    auto& gate = chain->gates.back();
    advance(tk_Identifier);
    
    // attribute
    auto attr = parseAttribute();
    if (attr)
      gate.attribute = std::move(attr);
    
    // target qubits
    while (curToken.is(tk_Numeric) || curToken.is(tk_All)) {
      if (curToken.is(tk_Numeric)) {
        if (!curToken.convertibleToInt()) {
          logErrHere("Expect a number for target qubit");
          failAndExit();
        }
        gate.qubits.emplace_back(
          std::make_unique<ast::SimpleNumericExpr>(curToken.toInt()));
        advance(tk_Numeric);
      }
      else {
        assert(curToken.is(tk_All));
        advance(tk_All);
        gate.qubits.emplace_back(std::make_unique<ast::AllExpr>());
      }
      if (curToken.is(tk_Comma))
        advance(tk_Comma);
    }
    if (gate.qubits.empty()) {
      logErrHere("Expect at least one target qubit");
      failAndExit();
    }
  };

  while (true) {
    requireCurTokenIs(tk_Identifier, "Expect a gate name");
    appendGateApplyStmt();
    if (curToken.is(tk_AtSymbol)) {
      advance(tk_AtSymbol);
      continue;
    }
    break;
  }
  requireCurTokenIs(tk_Semicolon, "Expect ';' to finish a GateChain");
  advance(tk_Semicolon);
  return chain;
}

ast::RootNode Parser::parse() {
  ast::RootNode root;
  while (curToken.isNot(tk_Eof)) {
    switch (curToken.kind) {
      case tk_Circuit: {
        root.stmts.emplace_back(std::move(parseCircuitStmt()));
        break;
      }
      default: {
        logErrHere("Expecting a top-level expression, which could be a "
                   "Circuit or Channel.");
        failAndExit();
      }
    }
  }
  return root;
}