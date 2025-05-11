#include "new_parser/Lexer.h"
#include <fstream>
#include <cassert>
#include "utils/iocolor.h"

using namespace cast::draft;

std::string cast::draft::internal::getNameOfTokenKind(TokenKind kind) {
  switch (kind) {
  case tk_Eof:
    return "EoF";
  case tk_Numeric:
    return "Numeric";
  case tk_Identifier:
    return "Identifier";

  case tk_Pi:
    return "pi";
  case tk_Circuit:
    return "circuit";

  case tk_L_RoundBracket:
    return "(";
  case tk_R_RoundBracket:
    return ")";
  case tk_L_CurlyBracket:
    return "{";
  case tk_R_CurlyBracket:
    return "}";

  case tk_Less:
    return "<";
  case tk_Greater:
    return ">";
  case tk_LessEqual:
    return "<=";
  case tk_GreaterEqual:
    return ">=";

  case tk_Comma:
    return ",";
  case tk_Semicolon:
    return ";";

  default:
    return "Unimplemented Name of TokenKind " +
           std::to_string(static_cast<int>(kind));
  }
}

std::ostream& Token::print(std::ostream& os) const {
  os << "tok(";

  if (kind == tk_Numeric) {
    os << "Num,";
    return os.write(loc.begin, length()) << ")";
  }

  if (kind == tk_Identifier) {
    os << "Identifier,";
    return os.write(loc.begin, length()) << ")";
  }

  os << IOColor::CYAN_FG;
  switch (kind) {
  case tk_Unknown:
    os << "Unknown";
    break;
  case tk_Eof:
    os << "EoF";
    break;
  case tk_L_RoundBracket:
    os << "(";
    break;
  case tk_R_RoundBracket:
    os << ")";
    break;
  case tk_L_SquareBracket:
    os << "[";
    break;
  case tk_R_SquareBracket:
    os << "]";
    break;
  case tk_L_CurlyBracket:
    os << "{";
    break;
  case tk_R_CurlyBracket:
    os << "}";
    break;
  case tk_Less:
    os << "<";
    break;
  case tk_Greater:
    os << ">";
    break;

  case tk_Comma:
    os << ",";
    break;
  case tk_Semicolon:
    os << ";";
    break;
  case tk_Percent:
    os << "%";
    break;
  case tk_AtSymbol:
    os << "@";
    break;

  default:
    os << static_cast<int>(kind) << " Not Imp'ed";
    break;
  }

  return os << IOColor::RESET << ")";
}

void Lexer::lexTwoChar(Token& tok, char snd, TokenKind tk1, TokenKind tk2) {
  if (*(curPtr + 1) == snd) {
    tok = Token(tk2, curPtr, curPtr + 2);
    curPtr += 2;
    return;
  }
  tok = Token(tk1, curPtr, curPtr + 1);
  ++curPtr;
}

void Lexer::lex(Token& tok) {
  if (curPtr >= sm.bufferEnd) {
    lexOneChar(tok, tk_Eof);
    --curPtr; // keep curPtr to its current position
    return;
  }

  char c = *curPtr;
  while (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
    c = *(++curPtr);
  }

  switch (c) {
  case '\0':
    lexOneChar(tok, tk_Eof);
    return;
  case '(':
    lexOneChar(tok, tk_L_RoundBracket);
    return;
  case ')':
    lexOneChar(tok, tk_R_RoundBracket);
    return;
  case '[':
    lexOneChar(tok, tk_L_SquareBracket);
    return;
  case ']':
    lexOneChar(tok, tk_R_SquareBracket);
    return;
  case '{':
    lexOneChar(tok, tk_L_CurlyBracket);
    return;
  case '}':
    lexOneChar(tok, tk_R_CurlyBracket);
    return;
  case ',':
    lexOneChar(tok, tk_Comma);
    return;
  case ';':
    lexOneChar(tok, tk_Semicolon);
    return;
  case '%':
    lexOneChar(tok, tk_Percent);
    return;
  case '#':
    lexOneChar(tok, tk_Hash);
    return;
  case '$':
    lexOneChar(tok, tk_Dollar);
    return;
  case '&':
    lexOneChar(tok, tk_Ampersand);
    return;
  case '|':
    lexOneChar(tok, tk_Pipe);
    return;
  case '^':
    lexOneChar(tok, tk_Caret);
    return;
  case '\'':
    lexOneChar(tok, tk_SingleQuote);
    return;
  case '"':
    lexOneChar(tok, tk_DoubleQuote);
    return;
  case '\\':
    lexOneChar(tok, tk_Backslash);
    return;
  case '`':
    lexOneChar(tok, tk_Backtick);
    return;
  case '!':
    lexOneChar(tok, tk_Exclamation);
    return;
  case '~':
    lexOneChar(tok, tk_Tilde);
    return;
  case ':':
    lexOneChar(tok, tk_Colon);
    return;
  case '?':
    lexOneChar(tok, tk_Question);
    return;
  case '.':
    lexOneChar(tok, tk_Dot);
    return;
  case '@':
    lexOneChar(tok, tk_AtSymbol);
    return;
  case '+':
    lexOneChar(tok, tk_Add);
    return;
  case '-':
    lexOneChar(tok, tk_Sub);
    return;

  // '<' or '<='
  case '<': {
    lexTwoChar(tok, '=', tk_Less, tk_LessEqual);
    return;
  }
  // '>' or '>='
  case '>': {
    lexTwoChar(tok, '=', tk_Greater, tk_GreaterEqual);
    return;
  }
  // '=' or '=='
  case '=': {
    lexTwoChar(tok, '=', tk_Equal, tk_EqualEqual);
    return;
  }
  // '*' or '**'
  case '*': {
    lexTwoChar(tok, '*', tk_Mul, tk_Pow);
    return;
  }
  case '/': {
    if (*(curPtr + 1) != '/') {
      lexOneChar(tok, tk_Div);
      return;
    }
    // we hit a comment, skip the rest of the line
    skipLine();
    return lex(tok);
  }

  default:
    auto* memRefBegin = curPtr;
    if ('0' <= c && c <= '9') {
      c = *(++curPtr);
      while (c == 'e' || c == '+' || c == '-' || c == '.' ||
             ('0' <= c && c <= '9'))
        c = *(++curPtr);
      tok = Token(tk_Numeric, memRefBegin, curPtr);
      return;
    }

    if (!std::isalpha(c)) {
      std::cerr << RED("[Lexer Error]: ") << "Unknown char "
                << static_cast<int>(c)
                << ". This is likely not implemented yet.\n";
      sm.printLineInfo(std::cerr, {curPtr, curPtr + 1});
      assert(false);
    }
    c = *(++curPtr);
    while (c == '_' || std::isalnum(c))
      c = *(++curPtr);

    tok = Token(tk_Identifier, memRefBegin, curPtr);
    // check for keywords
    auto tokStr = tok.toStringView();
    if (tokStr == "Pi")
      tok.kind = tk_Pi;
    else if (tokStr == "Circuit")
      tok.kind = tk_Circuit;
    else if (tokStr == "Channel")
      tok.kind = tk_Channel;
    else if (tokStr == "If")
      tok.kind = tk_If;
    else if (tokStr == "All")
      tok.kind = tk_All;
    else if (tokStr == "Measure")
      tok.kind = tk_Measure;
    else if (tokStr == "Repeat")
      tok.kind = tk_Repeat;

    return;
  }
}

void Lexer::skipLine() {
  while (curPtr < sm.bufferEnd) {
    if (*curPtr++ == '\n')
      break;
  }
}