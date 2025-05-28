#include "cast/Legacy/AST.h"
#include "cast/Legacy/Parser.h"
#include "cast/Legacy/CircuitGraph.h"

#include "utils/iocolor.h"

using namespace cast::legacy;
using namespace cast::legacy::ast;
using namespace IOColor;

std::string cast::legacy::getNameOfTokenKind(TokenKind kind) {
  switch (kind) {
  case tk_Eof:
    return "EoF";
  case tk_LineFeed:
    return "\\n";
  case tk_Numeric:
    return "Numeric";
  case tk_Identifier:
    return "Identifier";

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

Lexer::Lexer(const char* fileName) {
  std::ifstream file(fileName, std::ifstream::binary);
  assert(file);
  assert(file.is_open());

  file.seekg(0, file.end);
  bufferLength = file.tellg();
  file.seekg(0, file.beg);

  bufferBegin = new char[bufferLength];
  bufferEnd = bufferBegin + bufferLength;
  file.read(const_cast<char*>(bufferBegin), bufferLength);
  file.close();

  curPtr = bufferBegin;
  line = 1;
  lineBegin = bufferBegin;
}

void Parser::printLocation(std::ostream& os) const {
  auto lineInfo = lexer.getCurLineInfo();
  os << std::setw(5) << std::setfill(' ') << lineInfo.line << " | ";
  os.write(lineInfo.memRefBegin, lineInfo.memRefEnd - lineInfo.memRefBegin);
  os << "      | "
     << std::string(curToken.memRefBegin - lexer.lineBegin,' ')
     << BOLDGREEN(std::string(curToken.length(), '^') << "\n");
}

std::ostream& Token::print(std::ostream& os) const {
  os << "tok(";

  if (kind == tk_Numeric) {
    assert(memRefBegin != memRefEnd);
    os << "Num,";
    return os.write(memRefBegin, memRefEnd - memRefBegin) << ")";
  }

  if (kind == tk_Identifier) {
    assert(memRefBegin != memRefEnd);
    os << "Identifier,";
    return os.write(memRefBegin, memRefEnd - memRefBegin) << ")";
  }

  os << IOColor::CYAN_FG;
  switch (kind) {
  case tk_Unknown:
    os << "Unknown";
    break;
  case tk_Eof:
    os << "EoF";
    break;
  case tk_LineFeed:
    os << "\\n";
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
  if (curPtr >= bufferEnd) {
    lexOneChar(tok, tk_Eof);
    --curPtr; // keep curPtr to its current position
    return;
  }

  char c = *curPtr;
  while (c == ' ' || c == '\r')
    c = *(++curPtr);

  switch (c) {
  case '\0':
    lexOneChar(tok, tk_Eof);
    return;
  case '\n': {
    lexOneChar(tok, tk_LineFeed);
    ++line;
    lineBegin = curPtr;
    return;
  }

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

  case ',':
    lexOneChar(tok, tk_Comma);
    return;
  case ';':
    lexOneChar(tok, tk_Semicolon);
    return;
  case '%':
    lexOneChar(tok, tk_Percent);
    return;
  case '@':
    lexOneChar(tok, tk_AtSymbol);
    return;

  // '*' or '**'
  case '*': {
    lexTwoChar(tok, '*', tk_Mul, tk_Pow);
    return;
  }

  default:
    auto* memRefBegin = curPtr;
    if (c == '-' || ('0' <= c && c <= '9')) {
      c = *(++curPtr);
      while (c == 'e' || c == '+' || c == '-' || c == '.' ||
             ('0' <= c && c <= '9'))
        c = *(++curPtr);
      tok = Token(tk_Numeric, memRefBegin, curPtr);
      return;
    }

    if (!std::isalpha(c)) {
      auto lineInfo = getCurLineInfo();
      std::cerr << RED("[Lexer Error]: ") << "Unknown char "
                << static_cast<int>(c) << " at line " << lineInfo.line
                << ". This is likely not implemented yet.\n";
      assert(false);
    }
    c = *(++curPtr);
    while (c == '_' || std::isalnum(c))
      c = *(++curPtr);
    tok = Token(tk_Identifier, memRefBegin, curPtr);
    return;
  }
}

void Lexer::skipLine() {
  while (curPtr < bufferEnd) {
    if (*curPtr++ == '\n') {
      ++line;
      lineBegin = curPtr;
      break;
    }
  }
}

Lexer::LineInfo Lexer::getCurLineInfo() const {
  auto* lineEnd = curPtr;
  while (lineEnd < bufferEnd) {
    if (*lineEnd++ == '\n')
      break;
  }
  return { .line = line, .memRefBegin = lineBegin, .memRefEnd = lineEnd };
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
      logErr() << "Expect '+' or '-' when parsing a general complex number";
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

  logErr() << "Unable to parse complex number\n";
  return 0.0;
}

GateApplyStmt Parser::parseGateApply() {
  requireCurTokenIs(tk_Identifier);
  GateApplyStmt stmt(std::string(curToken.memRefBegin, curToken.memRefEnd), {});
  advance(tk_Identifier);

  // parameters (optional)
  if (optionalAdvance(tk_L_RoundBracket)) {
    if (optionalAdvance(tk_Hash)) {
      // parameter ref #N
      requireCurTokenIs(tk_Numeric);
      stmt.argument = curToken.toInt();
      advance(tk_Numeric);
    } else {
      // up to 3 parameters
      stmt.argument.set<legacy::GateMatrix::gate_params_t>();
      auto& parameters = stmt.argument.get<legacy::GateMatrix::gate_params_t>();
      for (int i = 0; i < 3; ++i) {
        if (curToken.is(tk_Numeric)) {
          parameters[i] = curToken.toDouble();
        } else if (optionalAdvance(tk_Percent)) {
          requireCurTokenIs(tk_Numeric);
          parameters[i] = curToken.toInt();
        } else {
          for (int j = i; j < 3; ++j)
            parameters[j].reset();
          break;
        }
        advance();
        optionalAdvance(tk_Comma);
      }
    }
    requireCurTokenIs(tk_R_RoundBracket);
    advance(tk_R_RoundBracket);
  }

  // target qubits
  while (true) {
    if (curToken.is(tk_Numeric)) {
      stmt.qubits.push_back(curToken.toInt());
      advance(tk_Numeric);
      optionalAdvance(tk_Comma);
      continue;
    }
    break;
  }

  // std::cerr << "Parsed gate " << stmt.name << " with "
  //           << stmt.qubits.size() << " targets, "
  //           << "curToken is " << getNameOfTokenKind(curToken.kind)
  //           << "\n";
  return stmt;
}

GateChainStmt Parser::parseGateChain() {
  GateChainStmt stmt;
  while (true) {
    stmt.gates.push_back(parseGateApply());
    if (optionalAdvance(tk_Semicolon))
      return stmt;
    if (optionalAdvance(tk_AtSymbol))
      continue;
    logErr() << "Unexpected token '"
             << getNameOfTokenKind(curToken.kind)
             << "' Did you forget ';'?\n";
    return stmt;
  }
}

QuantumCircuit Parser::parseQuantumCircuit() {
  if (curToken.isNot(tk_Identifier) || curToken.toStringView() != "circuit") {
    printLocation(
      logErr() << "Expect 'circuit' keyword in parsing QuantumCircuit\n");
    failAndExit();
  }
  advance(tk_Identifier);

  QuantumCircuit circuit;
  if (optionalAdvance(tk_Less)) {
    if (curToken.is(tk_Identifier) && curToken.toStringView() == "nQubits") {
      advance(tk_Identifier);
      advance(tk_Equal);
      requireCurTokenIs(tk_Numeric, "nQubits should be an integer");
      circuit.nQubits = curToken.toInt();
      advance(tk_Numeric);
      optionalAdvance(tk_Comma);
    }
    if (curToken.is(tk_Identifier) && curToken.toStringView() == "nParams") {
      advance(tk_Identifier);
      advance(tk_Equal);
      requireCurTokenIs(tk_Numeric, "nParams should be an integer");
      circuit.nParams = curToken.toInt();
      advance(tk_Numeric);
      optionalAdvance(tk_Comma);
    }
    requireCurTokenIs(tk_Greater);
    advance(tk_Greater);
  }

  requireCurTokenIs(tk_Identifier, "Expecting a name");
  circuit.name = curToken.toString();
  advance(tk_Identifier);
  skipLineBreaks();

  requireCurTokenIs(tk_L_CurlyBracket);
  advance(tk_L_CurlyBracket);
  skipLineBreaks();

  // circuit body
  while (true) {
    if (optionalAdvance(tk_R_CurlyBracket))
      break;
    circuit.addChainStmt(std::make_unique<GateChainStmt>(parseGateChain()));
    skipLineBreaks();
  }

  return circuit;
}
