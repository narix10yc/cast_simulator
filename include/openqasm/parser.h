#ifndef OPENQASM_PARSER_H
#define OPENQASM_PARSER_H

#include "openqasm/AST.h"
#include "openqasm/Lexer.h"
#include <memory>
#include <queue>
#include <string>

namespace openqasm {

class Parser {
  int debugLevel;
  std::unique_ptr<Lexer> lexer;
  std::queue<Token> tokenBuf;
  Token curToken;

public:
  Parser(const std::string& fileName, int debugLevel = 0)
      : debugLevel(debugLevel), lexer(std::make_unique<Lexer>(fileName)),
        curToken(TokenTy::Unknown) {}

  void logError(std::string str) const {
    fprintf(stderr, "== Parser Error ==  %s\n", str.c_str());
  }

  void logDebug(int level, std::string msg) const {
    if (debugLevel < level)
      return;
    std::cerr << std::string(" ", level) << "[Parser DEBUG " << level << "] "
              << msg << "\n";
  }

  void nextToken() {
    if (tokenBuf.empty())
      curToken = lexer->getToken();
    else {
      curToken = tokenBuf.front();
      tokenBuf.pop();
    }
  }

  bool expectNextToken(TokenTy type, std::string msg = "") {
    nextToken();
    if (curToken.type != type) {
      logError("Expect token with type " + tokenTypetoString(type) + msg);
      return false;
    }
    return true;
  }

  void skipSeparators() {
    while (curToken.type == TokenTy::Semicolon ||
           curToken.type == TokenTy::LineFeed ||
           curToken.type == TokenTy::CarriageReturn) {
      nextToken();
    }
  }

  /*
      return the next token;
      curToken does not change.
  */
  Token peek() {
    Token curTokenCopy = curToken;
    nextToken();
    Token nextTokenCopy = curToken;
    tokenBuf.push(nextTokenCopy);
    curToken = curTokenCopy;
    return nextTokenCopy;
  }

  /*
      The entry point of the parsing process. This will update the root
      variable
  */
  std::unique_ptr<ast::RootNode> parse();

private:
  /*
      Parse a statement.
      After this method returns, curToken is always at the start
      of the next statement.
  */
  std::unique_ptr<ast::Statement> parseStmt();

  std::unique_ptr<ast::IfThenElseStmt> parseIfThenElseStmt();

  std::unique_ptr<ast::VersionStmt> parseVersionStmt();

  std::unique_ptr<ast::QRegStmt> parseQRegStmt();

  std::unique_ptr<ast::CRegStmt> parseCRegStmt();

  std::unique_ptr<ast::GateApplyStmt> parseGateApplyStmt();

  /*
      Call this when curToken is at the first Token in an expression
      After this methods returns, curToken will be at the next Token
  */
  std::unique_ptr<ast::Expression> parseExpr();

  std::unique_ptr<ast::Expression>
  parseExprRHS(BinaryOp lhsBinop, std::unique_ptr<ast::Expression>&& lhs);

  /*
      There are 3 types of primary expr:
      - numerics
      - variable (including funcCall and subscipt)
      - parenthesis
  */
  std::unique_ptr<ast::Expression> parsePrimaryExpr();

  std::unique_ptr<ast::NumericExpr> parseNumericExpr();

  /* Call this when curToken is an identifier */
  std::unique_ptr<ast::Expression> parseVariableExpr();

  std::unique_ptr<ast::SubscriptExpr> parseSubscriptExpr();

  std::unique_ptr<ast::Expression> parseParenExpr();
};

} // namespace openqasm

#endif // OPENQASM_PARSER_H