#ifndef NEW_PARSER_LEXER_H
#define NEW_PARSER_LEXER_H

#include "new_parser/SourceManager.h"
#include <string>
#include <cassert>

namespace cast::draft {

enum TokenKind : int {
  tk_Eof = -1,
  tk_Identifier = -2,
  tk_Numeric = -3,

  // keywords
  tk_Pi = -10,
  tk_Circuit = -11,
  tk_Channel = -12,
  tk_Measure = -13,
  tk_If = -14,
  tk_Else = -15,
  tk_All = -16,
  tk_Repeat = -17,

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
  tk_Comma              = -104,  // ,
  tk_Semicolon          = -105,  // ;
  tk_L_RoundBracket     = -106,  // (
  tk_R_RoundBracket     = -107,  // )
  tk_L_SquareBracket    = -108,  // [
  tk_R_SquareBracket    = -109,  // ]
  tk_Colon              = -110,  // :
  tk_L_CurlyBracket     = -112,  // {
  tk_R_CurlyBracket     = -113,  // }
  tk_SingleQuote        = -114,  // '
  tk_DoubleQuote        = -115,  // "
  tk_AtSymbol           = -116,  // @
  tk_Percent            = -117,  // %
  tk_Hash               = -118,  // #
  tk_Backslash          = -119,  // '\'
  tk_Slash              = -120,  // /
  tk_Tilde              = -121,  // ~
  tk_Exclamation        = -122,  // !
  tk_Question           = -123,  // ?
  tk_Ampersand          = -124,  // &
  tk_Pipe               = -125,  // |
  tk_Caret              = -126,  // ^
  tk_Dot                = -133,  // .
  tk_Dollar             = -134,  // $
  tk_Backtick           = -135,  // `

  tk_Unknown = -1000,
  tk_Any = -1001,
};

namespace internal {
  std::string getNameOfTokenKind(TokenKind);
} // namespace internal

class Token {
public:
  TokenKind kind;
  LocationSpan loc;

  Token(TokenKind kind = tk_Unknown)
      : kind(kind), loc(nullptr, nullptr) {}

  Token(TokenKind kind, const char* memRefBegin, const char* memRefEnd)
      : kind(kind), loc(memRefBegin, memRefEnd) {}

  std::ostream& print(std::ostream& = std::cerr) const;

  bool is(TokenKind k) const { return kind == k; }
  bool isNot(TokenKind k) const { return kind != k; }

  // is the token the literal 'i'
  bool isI() const {
    return kind == tk_Identifier && length() == 1 && *loc.begin == 'i';
  }

  bool convertibleToInt() const {
    for (const char* p = loc.begin; p < loc.end; ++p) {
      if (!std::isdigit(*p))
        return false;
    }
    return true;
  }

  double toDouble() const {
    return std::stod(std::string(loc.begin, loc.end));
  }

  int toInt() const {
    assert(convertibleToInt());
    return std::stoi(std::string(loc.begin, loc.end));
  }

  std::string_view toStringView() const {
    return std::string_view(loc.begin, loc.end - loc.begin);
  }

  std::string toString() const {
    return std::string(loc.begin, loc.end);
  }

  size_t length() const { return loc.end - loc.begin; }
};

class Lexer {
private:
  /// Set \c tok to TokenKind \c tk and range [curPtr, curPtr + 1)
  /// Set curPtr to curPtr + 1 
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
  SourceManager sm;
  const char* curPtr;

  Lexer() : sm(), curPtr(nullptr) {}

  // return true on error
  bool loadFromFile(const char* filename) {
    if (sm.loadFromFile(filename))
      return true; // error loading file
    curPtr = sm.bufferBegin;
    return false; // success
  }

  // return true on error
  bool loadRawBuffer(const char* buffer, size_t size) {
    if (sm.loadRawBuffer(buffer, size))
      return true; // error loading buffer
    curPtr = sm.bufferBegin;
    return false; // success
  }

  /// After this function returns, curPtr always points to the next char after 
  /// \c tok
  void lex(Token& tok);

  void skipLine();
};

} // namespace cast::draft

#endif // NEW_PARSER_LEXER_H