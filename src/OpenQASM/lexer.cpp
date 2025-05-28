#include "openqasm/lexer.h"

using namespace openqasm;

std::string Token::toString() const {
  if (type == TokenTy::Eof) {
    return "EOF";
  }
  if (type == TokenTy::If) {
    return "if";
  }
  if (type == TokenTy::Else) {
    return "else";
  }
  if (type == TokenTy::Then) {
    return "then";
  }

  if (type == TokenTy::Unknown) {
    return "unknown";
  }

  if (type == TokenTy::Numeric)
    return "Numeric(" + str + ")";

  if (type == TokenTy::Identifier)
    return "Identifier(" + str + ")";

  return std::to_string(static_cast<int>(type));
}

int Lexer::peekChar() {
  int curCharCopy = curChar;
  nextChar();
  int nextCharCopy = curChar;
  charBuf.push(nextCharCopy);
  curChar = curCharCopy;
  return nextCharCopy;
}

Token Lexer::getToken() {
  // skip spaces
  if (!waitFlag)
    nextChar();

  while (curChar == ' ')
    nextChar();

  waitFlag = false;
  // numerics
  if (std::isdigit(curChar) || curChar == '.')
    return tokenizeNumeric();
  if (std::isalpha(curChar) || curChar == '_')
    return tokenizeIdentifier();

  if (curChar == '#') { // comments
    skipToEndOfLine();
  }

  if (curChar == '/') {
    int next = peekChar();
    if (next == '/')
      skipToEndOfLine();
    else {
      return TokenTy::Div;
    }
  }

  switch (curChar) {
  case '+':
    return TokenTy::Add;
  case '-':
    return TokenTy::Sub;
  case '*':
    return TokenTy::Mul;

  case ',':
    return TokenTy::Comma;
  case ';':
    return TokenTy::Semicolon;
  case '(':
    return TokenTy::L_RoundBracket;
  case ')':
    return TokenTy::R_RoundBracket;
  case '[':
    return TokenTy::L_SquareBracket;
  case ']':
    return TokenTy::R_SquareBracket;
  case '{':
    return TokenTy::L_CurlyBracket;
  case '}':
    return TokenTy::R_CurlyBracket;
  case '<':
    return TokenTy::L_AngleBracket;
  case '>':
    return TokenTy::R_AngleBracket;
  case '\'':
    return TokenTy::SingleQuote;
  case '\"':
    return TokenTy::DoubleQuote;

  case '\n':
    return TokenTy::LineFeed;
  case '\r':
    return TokenTy::CarriageReturn;

  case EOF:
    return TokenTy::Eof;
  default:
    return TokenTy::Unknown;
  }
}

Token Lexer::tokenizeNumeric() {
  waitFlag = true;
  std::string numericStr;
  do {
    numericStr += curChar;
    nextChar();
  } while (std::isdigit(curChar) || curChar == '.' || curChar == 'e' ||
           curChar == '-');

  // check number of dots
  int count = 0;
  for (char c : numericStr) {
    if (c == '.') {
      count++;
      if (count == 2) {
        logError("Invalid numerics '" + numericStr + "'");
        return {TokenTy::Unknown};
      }
    }
  }

  return {TokenTy::Numeric, numericStr};
}

Token Lexer::tokenizeIdentifier() {
  waitFlag = true;
  std::string identifier = "";
  do {
    identifier += curChar;
    nextChar();
  } while (std::isalnum(curChar) || curChar == '_');

  if (identifier == "if")
    return TokenTy::If;
  if (identifier == "else")
    return TokenTy::Else;
  if (identifier == "OPENQASM")
    return TokenTy::Openqasm;
  if (identifier == "include")
    return TokenTy::Include;
  if (identifier == "qreg")
    return TokenTy::Qreg;
  if (identifier == "creg")
    return TokenTy::Creg;
  if (identifier == "gate")
    return TokenTy::Gate;

  return {TokenTy::Identifier, identifier};
}
