#include "openqasm/parser.h"

using namespace openqasm;

std::unique_ptr<ast::IfThenElseStmt> Parser::parseIfThenElseStmt() {
  logDebug(2, "IfThenElseStmt: ready to parse");
  nextToken(); // eat 'if'
  auto ifExpr = parseExpr();
  if (!ifExpr) {
    logError("IfThenElseStmt: failed to parse if expression");
    return nullptr;
  }

  auto ifThenElseStmr =
      std::make_unique<ast::IfThenElseStmt>(std::move(ifExpr));

  // parse then block
  if (curToken.type != TokenTy::L_CurlyBracket) {
    ifThenElseStmr->addThenBody(parseStmt());
    logDebug(2, "IfThenElseStmt: thenBody successfully parsed");
  } else {
    nextToken(); // eat '{'
    while (curToken.type != TokenTy::R_CurlyBracket) {
      auto item = parseStmt();
      if (!item) {
        logError("IfThenElseStmt: cannot parse thenBody stmt");
        return nullptr;
      }
      logDebug(2, "IfThenElseStmt: parsed thenBody stmt " + item->toString());
      ifThenElseStmr->addThenBody(std::move(item));
    }
    nextToken(); // eat '}'
    logDebug(2, "IfThenElseStmt: thenBody successfully parsed");
  }

  // parse else block
  if (curToken.type != TokenTy::Else) {
    logDebug(2, "IfThenElseStmt: success (no parse body)");
    return ifThenElseStmr;
  }

  nextToken(); // eat 'else'
  if (curToken.type != TokenTy::L_CurlyBracket) {
    ifThenElseStmr->addElseBody(parseStmt());
    logDebug(2, "IfThenElseStmt: elseBody successfully parsed");
  } else {
    nextToken(); // eat '{'
    while (curToken.type != TokenTy::R_CurlyBracket) {
      auto item = parseStmt();
      if (!item) {
        logError("IfThenElseStmt: cannot parse elseBody stmt");
        return nullptr;
      }
      logDebug(2, "IfThenElseStmt: parsed elseBody stmt " + item->toString());
      ifThenElseStmr->addElseBody(std::move(item));
    }
    nextToken(); // eat '}'
    logDebug(2, "IfThenElseStmt: thenBody successfully parsed");
  }

  return ifThenElseStmr;
}