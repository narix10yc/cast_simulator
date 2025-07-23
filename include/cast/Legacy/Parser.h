#ifndef CAST_LEGACY_PARSER_H
#define CAST_LEGACY_PARSER_H

#include "cast/Legacy/AST.h"

#include <cassert>
#include <complex>
#include <fstream>
#include <iostream>
#include <string_view>

#include "utils/iocolor.h"

namespace cast::legacy {
class CircuitGraph;
}

namespace cast::legacy {

enum TokenKind : int {
  tk_Eof = -1,
  tk_Identifier = -2,
  tk_Numeric = -3,

  // keywords
  tk_Circuit = -10,

  // operators
  tk_Add = -30,          // +
  tk_Sub = -31,          // -
  tk_Mul = -32,          // *
  tk_Div = -33,          // /
  tk_Pow = -34,          // **
  tk_Greater = -35,      // >
  tk_Less = -36,         // <
  tk_Equal = -37,        // =
  tk_GreaterEqual = -38, // >=
  tk_LessEqual = -39,    // <=
  tk_EqualEqual = -40,   // ==

  // symbols
  tk_Comma = -104,           // ,
  tk_Semicolon = -105,       // ;
  tk_L_RoundBracket = -106,  // (
  tk_R_RoundBracket = -107,  // )
  tk_L_SquareBracket = -108, // [
  tk_R_SquareBracket = -109, // ]
  tk_L_CurlyBracket = -112,  // {
  tk_R_CurlyBracket = -113,  // }
  tk_SingleQuote = -114,     // '
  tk_DoubleQuote = -115,     // "
  tk_AtSymbol = -116,        // @
  tk_Percent = -117,         // %
  tk_Hash = -118,            // #
  tk_Backslash = -119,       // '\'
  tk_Comment = -120,         // '//'
  tk_CommentStart = -121,    // '/*'
  tk_CommentEnd = -122,      // '*/'

  tk_LineFeed = 10,       // '\n'
  tk_CarriageReturn = 13, // '\r'

  tk_Unknown = -1000,
  tk_Any = -1001,
};

std::string getNameOfTokenKind(TokenKind);

class Token {
public:
  TokenKind kind;
  const char* memRefBegin;
  const char* memRefEnd;

  Token(TokenKind kind = tk_Unknown)
      : kind(kind), memRefBegin(nullptr), memRefEnd(nullptr) {}

  Token(TokenKind kind, const char* memRefBegin, const char* memRefEnd)
      : kind(kind), memRefBegin(memRefBegin), memRefEnd(memRefEnd) {}

  std::ostream& print(std::ostream& = std::cerr) const;

  bool is(TokenKind k) const { return kind == k; }
  bool isNot(TokenKind k) const { return kind != k; }

  // is the token the literal 'i'
  bool isI() const {
    return kind == tk_Identifier && length() == 1 && *memRefBegin == 'i';
  }

  double toDouble() const {
    assert(memRefBegin < memRefEnd);
    return std::stod(std::string(memRefBegin, memRefEnd));
  }

  int toInt() const {
    assert(memRefBegin < memRefEnd);
    return std::stoi(std::string(memRefBegin, memRefEnd));
  }

  std::string_view toStringView() const {
    assert(memRefBegin < memRefEnd);
    return std::string_view(memRefBegin, memRefEnd);
  }

  std::string toString() const {
    assert(memRefBegin < memRefEnd);
    return std::string(memRefBegin, memRefEnd);
  }

  size_t length() const { return memRefEnd - memRefBegin; }
};

class Lexer {
private:
  /// lex a token with a single char.
  /// After this function returns, curPtr always points to the next char after
  /// \c tok
  void lexOneChar(Token& tok, TokenKind tk) {
    tok = Token(tk, curPtr, curPtr + 1);
    ++curPtr;
  }

  /// lex a token with possibly two chars. If *(curPtr + 1) matches snd, \c tok
  /// is assigned with TokenKind \c tk2. Otherwise, \c tok is assigned with
  /// TokenKind \c tk1.
  /// When calling this function, curPtr should point to the first char of this
  /// token. For example, lexTwoChar(tok, '=', tk_Less, tk_LessEqual) should be
  /// called when curPtr points to '<', and it conditionally checks if curPtr+1
  /// points to '='.
  /// After this function returns, curPtr always points to the next char after
  /// \c tok
  void lexTwoChar(Token& tok, char snd, TokenKind tk1, TokenKind tk2);

public:
  const char* bufferBegin;
  const char* bufferEnd;
  size_t bufferLength;

  const char* curPtr;

  int line;
  const char* lineBegin;

  explicit Lexer(const char* fileName);

  Lexer(const Lexer&) = delete;
  Lexer(Lexer&&) = delete;

  Lexer& operator=(const Lexer&) = delete;
  Lexer& operator=(Lexer&&) = delete;

  ~Lexer() { delete[] bufferBegin; }

  /// After this function returns, curPtr always points to the next char after
  /// \c tok
  void lex(Token& tok);

  void skipLine();

  struct LineInfo {
    int line;
    const char* memRefBegin;
    const char* memRefEnd;
  };

  LineInfo getCurLineInfo() const;
};

class Parser {
  Lexer lexer;

  Token curToken;
  Token nextToken;

  std::complex<double> parseComplexNumber();

public:
  Parser(const char* fileName) : lexer(fileName) {
    lexer.lex(curToken);
    lexer.lex(nextToken);
  }

  void printLocation(std::ostream& os = std::cerr) const;

  std::ostream& logErr() const {
    return std::cerr << BOLDRED("Parser Error: ");
  }

  void failAndExit() const {
    std::cerr << BOLDRED("Parsing failed. Exiting...\n");
    exit(1);
  }

  void skipLineBreaks() {
    while (curToken.is(tk_LineFeed))
      advance(tk_LineFeed);
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

  /// Advance such that curToken must have \p kind. Otherwise, terminate the
  /// program with error messages
  void requireCurTokenIs(TokenKind kind, const char* msg = nullptr) const {
    if (curToken.is(kind))
      return;
    auto& os = logErr();
    if (msg)
      os << msg;
    else
      os << "Requires a '" << getNameOfTokenKind(kind) << "' token";
    os << " (Got '" << getNameOfTokenKind(curToken.kind) << "')\n";
    printLocation(os);
    failAndExit();
  }

  ast::GateApplyStmt parseGateApply();
  ast::GateChainStmt parseGateChain();

  ast::QuantumCircuit parseQuantumCircuit();
};

} // namespace cast::legacy

#endif // CAST_LEGACY_PARSER_H