#ifndef SAOT_NEWPARSER_H
#define SAOT_NEWPARSER_H

#include <cassert>
#include <complex>
#include <fstream>
#include <iostream>

#include "utils/iocolor.h"

namespace saot {
class CircuitGraph;
}

namespace saot::ast {
class GateApplyStmt;
class GateChainStmt;
class QuantumCircuit;
} // namespace saot::ast

namespace saot::parse {

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
  tk_Comma = -104,          // ,
  tk_Semicolon = -105,      // ;
  tk_L_RoundBraket = -106,  // (
  tk_R_RoundBraket = -107,  // )
  tk_L_SquareBraket = -108, // [
  tk_R_SquareBraket = -109, // ]
  tk_L_CurlyBraket = -112,  // {
  tk_R_CurlyBraket = -113,  // }
  tk_SingleQuote = -114,    // '
  tk_DoubleQuote = -115,    // "
  tk_AtSymbol = -116,       // @
  tk_Percent = -117,        // %
  tk_Hash = -118,           // #
  tk_Backslash = -119,      // '\'
  tk_Comment = -120,        // '//'
  tk_CommentStart = -121,   // '/*'
  tk_CommentEnd = -122,     // '*/'

  tk_LineFeed = 10,       // '\n'
  tk_CarriageReturn = 13, // '\r'

  tk_Unknown = -1000,
  tk_Any = -1001,
};

std::string getNameOfTokenKind(TokenKind);

class Token {
public:
  TokenKind kind;
  const char *memRefBegin;
  const char *memRefEnd;

  Token(TokenKind kind = tk_Unknown)
      : kind(kind), memRefBegin(nullptr), memRefEnd(nullptr) {}

  Token(TokenKind kind, const char *memRefBegin, const char *memRefEnd)
      : kind(kind), memRefBegin(memRefBegin), memRefEnd(memRefEnd) {}

  std::ostream &print(std::ostream & = std::cerr) const;

  bool is(TokenKind k) { return kind == k; }
  bool isNot(TokenKind k) { return kind != k; }

  // is the token the literal 'i'
  bool isI() {
    return kind == tk_Identifier && memRefBegin + 1 == memRefEnd &&
           *memRefBegin == 'i';
  }

  double toDouble() const {
    assert(memRefBegin < memRefEnd);
    return std::stod(std::string(memRefBegin, memRefEnd));
  }

  int toInt() const {
    assert(memRefBegin < memRefEnd);
    return std::stoi(std::string(memRefBegin, memRefEnd));
  }
};

class Lexer {
public:
  const char *bufferBegin;
  const char *bufferEnd;
  size_t bufferLength;

  const char *curPtr;

  int line;
  const char *lineBegin;

  Lexer(const char *fileName) {
    std::ifstream file(fileName, std::ifstream::binary);
    assert(file.is_open());

    file.seekg(0, file.end);
    bufferLength = file.tellg();
    file.seekg(0, file.beg);

    bufferBegin = new char[bufferLength];
    bufferEnd = bufferBegin + bufferLength;
    file.read(const_cast<char *>(bufferBegin), bufferLength);
    file.close();

    curPtr = bufferBegin;
    line = 1;
    lineBegin = bufferBegin;
  }

  void lex(Token &tok);

  void skipLine();

  struct line_info_t {
    int line;
    const char *memRefBegin;
    const char *memRefEnd;
  };

  line_info_t getCurLineInfo() const;
};

class Parser {
  Lexer lexer;

  Token curToken;
  Token nextToken;

  std::complex<double> parseComplexNumber();

public:
  Parser(const char *fileName) : lexer(fileName) {
    lexer.lex(curToken);
    lexer.lex(nextToken);
  }

  void printLocation(std::ostream &os = std::cerr) const;

  std::ostream &logErr() const {
    return std::cerr << IOColor::RED_FG << IOColor::BOLD
                     << "Parser Error: " << IOColor::RESET;
  }

  void failAndExit() const {
    std::cerr << IOColor::RED_FG << IOColor::BOLD
              << "Parsing failed. Exiting...\n"
              << IOColor::RESET;
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

  bool optionalAdvance(TokenKind kind) {
    if (curToken.is(kind)) {
      advance();
      return true;
    }
    return false;
  }

  void requiredAdvance(TokenKind kind, const char *msg = nullptr) {
    if (curToken.is(kind)) {
      advance();
      return;
    }

    auto &os = logErr();
    if (msg)
      os << msg;
    else
      os << "Require TokenKind '" << getNameOfTokenKind(kind) << "'";
    os << " (Got '" << getNameOfTokenKind(curToken.kind) << "')\n";
    printLocation(os);
    failAndExit();
  }

  void requireCurTokenIs(TokenKind kind, const char *msg = nullptr) {
    if (curToken.is(kind))
      return;
    auto &os = logErr();
    if (msg)
      os << msg;
    else
      os << "Require curToken is '" << getNameOfTokenKind(kind) << "'";
    os << " (Got '" << getNameOfTokenKind(curToken.kind) << "')\n";
    printLocation(os);
    failAndExit();
  }

  ast::GateApplyStmt parseGateApply();
  ast::GateChainStmt parseGateChain();

  ast::QuantumCircuit parseQuantumCircuit();
};

} // namespace saot::parse

#endif // SAOT_NEWPARSER_H