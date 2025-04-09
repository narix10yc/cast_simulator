#ifndef CAST_DRAFT_PARSER_H
#define CAST_DRAFT_PARSER_H

#include "new_parser/AST.h"
#include "new_parser/Lexer.h"
#include "utils/iocolor.h"
#include <complex>

namespace cast::draft {
class Parser {
  Lexer lexer;

  Token curToken;
  Token nextToken;

  std::complex<double> parseComplexNumber();

  // Parse attributes assocated with a statement. When invoked, it checks if 
  // curToken is '[', and if so, parse attribute list (ended with ']').
  // Otherwise it returns nullptr.
  std::unique_ptr<ast::Attribute> parseAttribute();

  std::unique_ptr<ast::CircuitStmt> parseCircuitStmt();
  
  // Expression-related
  std::unique_ptr<ast::Expr> parseExpr(int precedence = 0);
  std::unique_ptr<ast::Expr> parsePrimaryExpr();

  void printLocation(std::ostream& os = std::cerr) const;

  std::ostream& logErr() const {
    return std::cerr << BOLDRED("Parser Error: ");
  }

  void failAndExit() const {
    std::cerr << BOLDRED("Parsing failed. Exiting...\n");
    exit(1);
  }

  void advance() {
    curToken = nextToken;
    lexer.lex(nextToken);
  }

  void advance(TokenKind kind) {
    assert(curToken.is(kind) && "kind mismatch in 'advance'");
    advance();
  }

  /// If curToken matches \c kind, calls \c advance() and returns true;
  /// Otherwise nothing happens and returns false
  bool optionalAdvance(TokenKind kind) {
    if (curToken.is(kind)) {
      advance();
      return true;
    }
    return false;
  }

  /// Assert curToken must have \p kind. Otherwise, terminate the 
  /// program with error messages
  void requireCurTokenIs(TokenKind kind, const char* msg = nullptr) const;
public:
  Parser(const char* fileName) : lexer(fileName) {
    lexer.lex(curToken);
    lexer.lex(nextToken);
  }

  ast::RootNode parse();
};


} // namespace cast::draft

#endif // CAST_DRAFT_PARSER_H