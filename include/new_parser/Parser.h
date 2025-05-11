#ifndef CAST_DRAFT_PARSER_H
#define CAST_DRAFT_PARSER_H

#include "new_parser/ASTContext.h"
#include "new_parser/AST.h"
#include "new_parser/Lexer.h"
#include "utils/iocolor.h"
#include <complex>
#include <unordered_map>

namespace cast::draft {

class Scope {
private:
  struct IdentifierHash {
    std::size_t operator()(const ast::Identifier& id) const {
      return std::hash<std::string_view>()(id.str);
    }
  }; // struct IdentifierHash
public:
  std::unordered_map<ast::Identifier, ast::Node*, IdentifierHash> symbols;
  Scope* parent;
  explicit Scope(Scope* parent) : symbols(), parent(parent) {}
}; // class Scope

class Parser {
  ASTContext& ctx;
  Lexer lexer;
  Scope* currentScope;

  Token curToken;
  Token nextToken;

  std::complex<double> parseComplexNumber();

  // Parse attributes assocated with a statement. When invoked, it checks if 
  // curToken is '<', and if so, parse attribute list (ended with '>').
  // Otherwise it returns nullptr.
  ast::Attribute* parseAttribute();

  // Parse a parameter declaration. When invoked, it checks if curToken is '(',
  // and if so, parse parameter list (ended with ')').
  // Otherwise it returns nullptr.
  ast::ParameterDeclExpr* parseParameterDecl();

  // Used in \c parsePrimaryStmt, triggered by when \c curToken is an identfier
  ast::Expr* parseIdentifierOrCallExpr();
  
  // CircuitStmt is a top-level statement. Should only be called when curToken
  // is 'Circuit'. This function never returns nullptr.
  ast::CircuitStmt* parseCircuitStmt();

  // Circuit-level statements include GateChainStmt, MeasureStmt
  // Possibly returns nullptr
  ast::Stmt* parseCircuitLevelStmt();

  ast::GateChainStmt* parseGateChainStmt();
  // Never returns nullptr
  ast::GateApplyStmt* parseGateApplyStmt();

  // ChannelStmt is a top-level statement. Should only be called when curToken
  // is 'Channel'. Never returns nullptr.
  ast::ChannelStmt* parseChannelStmt();

  // Parse a PauliComponentStmt. Should not be called when curToken is an
  // identifier. Never returns nullptr.
  ast::PauliComponentStmt* parsePauliComponentStmt();
  
  // Possibly returns nullptr
  ast::Expr* parseExpr(int precedence = 0);

  // Possibly returns nullptr
  ast::Expr* parsePrimaryExpr();

  // Try to convert a general expression to a simple numeric expression.
  // Return nullptr if the conversion is not possible.
  ast::SimpleNumericExpr* convertExprToSimpleNumeric(ast::Expr* expr);

  void pushScope() {
    currentScope = new Scope(currentScope);
  }

  void popScope() {
    assert(currentScope && "Trying to pop a null scope");
    Scope* parent = currentScope->parent;
    delete currentScope;
    currentScope = parent;
  }

  void addSymbol(ast::Identifier name, ast::Node* node);

  ast::Node* lookup(ast::Identifier name);

  std::ostream& err() const {
    return std::cerr << BOLDRED("Parser Error: ");
  }

  std::ostream& printLineInfo(LocationSpan loc) const {
    return lexer.sm.printLineInfo(std::cerr, loc);
  }
  
  // A convenience function to print error message followed by line info.
  void logErrHere(const char* msg) const {
    err() << msg << "\n";
    lexer.sm.printLineInfo(std::cerr, curToken.loc);
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
  Parser(ASTContext& ctx, const char* fileName)
    : ctx(ctx), lexer(fileName), currentScope(nullptr) {
    lexer.lex(curToken);
    lexer.lex(nextToken);
  }

  Parser(const Parser&) = delete;
  Parser& operator=(const Parser&) = delete;
  Parser(Parser&&) = delete;
  Parser& operator=(Parser&&) = delete;

  ~Parser() {
    assert(currentScope == nullptr && "Parser exits with a non-empty scope");
  }

  ast::RootNode* parse();

  void displayLineTable() const;
};


} // namespace cast::draft

#endif // CAST_DRAFT_PARSER_H