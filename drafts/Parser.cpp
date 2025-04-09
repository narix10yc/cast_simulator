#include "new_parser/Parser.h"
#include <fstream>
#include <cassert>

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
  lineNumber = 1;
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
  while (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
    if (c == '\n') {
      ++lineNumber;
      lineBegin = curPtr + 1;
    }
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
    // check for keywords
    if (tok.toStringView() == "pi")
      tok.kind = tk_Pi;
    else if (tok.toStringView() == "circuit")
      tok.kind = tk_Circuit;
    return;
  }
}

void Lexer::skipLine() {
  while (curPtr < bufferEnd) {
    if (*curPtr++ == '\n') {
      ++lineNumber;
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
  return { .line = lineNumber, .memRefBegin = lineBegin, .memRefEnd = lineEnd };
}

void Parser::requireCurTokenIs(TokenKind kind, const char* msg) const {
  if (curToken.is(kind))
    return;
  auto& os = logErr();
  if (msg)
    os << msg;
  else
    os << "Requires a '" << internal::getNameOfTokenKind(kind) << "' token";
  os << " (Got '" << internal::getNameOfTokenKind(curToken.kind) << "')\n";
  printLocation(os);
  failAndExit();
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

// Try to convert a general expression to a simple numeric expression.
// This function is mainly used in parsing the 'phase' attribute of statements.
static std::unique_ptr<ast::SimpleNumericExpr>
convertExprToSimpleNumeric(const ast::Expr* expr) {
  if (expr == nullptr)
    return nullptr;
  if (auto* e = llvm::dyn_cast<ast::ParameterExpr>(expr))
    return nullptr;

  if (auto* e = llvm::dyn_cast<ast::SimpleNumericExpr>(expr))
    return std::make_unique<ast::SimpleNumericExpr>(*e);

  if (auto* e = llvm::dyn_cast<ast::BinaryOpExpr>(expr)) {
    auto lhs = convertExprToSimpleNumeric(e->lhs.get());
    auto rhs = convertExprToSimpleNumeric(e->rhs.get());
    if (!lhs || !rhs)
      return nullptr;
    switch (e->op) {
      case ast::BinaryOpExpr::Add:
        return std::make_unique<ast::SimpleNumericExpr>(*lhs + *rhs);
      case ast::BinaryOpExpr::Sub:
        return std::make_unique<ast::SimpleNumericExpr>(*lhs - *rhs);
      case ast::BinaryOpExpr::Mul:
        return std::make_unique<ast::SimpleNumericExpr>(*lhs * *rhs);
      case ast::BinaryOpExpr::Div:
        return std::make_unique<ast::SimpleNumericExpr>(*lhs / *rhs);
      default:
        assert(false && "Illegal binary operator");
        return nullptr;
    }
  }

  assert(false && "Not Implemented or illegal expression");
  return nullptr;

}

void Parser::parseAttribute(ast::Attribute& attr) {
  if (curToken.isNot(tk_L_SquareBracket))
    return;

  advance(tk_L_SquareBracket);
  while (curToken.isNot(tk_R_SquareBracket)) {
    requireCurTokenIs(tk_Identifier, "Attribute name expected");
    auto name = curToken.toString();
    advance(tk_Identifier);
    requireCurTokenIs(tk_Equal, "Expect '=' after attribute name");
    advance(tk_Equal);

    // 'nqubits', 'nparams', and 'phase' are reserved attributes
    if (name == "nqubits") {
      requireCurTokenIs(tk_Numeric, "Attribute 'nqubits' must be a number");
      attr.nQubits = curToken.toInt();
      advance(tk_Numeric);
    }
    else if (name == "nparams") {
      requireCurTokenIs(tk_Numeric, "Attribute 'nparams' must be a number");
      attr.nQubits = curToken.toInt();
      advance(tk_Numeric);
    }
    else if (name == "phase") {
      auto expr = parseExpr();
      if (expr == nullptr) {
        logErr() << "Failed to parse expression for attribute 'phase'\n";
        printLocation();
        failAndExit();
      }
      auto simpleNumeric = convertExprToSimpleNumeric(expr.get());
      if (simpleNumeric == nullptr) {
        auto& os = logErr();
        os << "Expression ";
        expr->print(os) << " cannot be used in attribute 'phase'\n";
        printLocation();
        failAndExit();
      }
      attr.phase = *simpleNumeric;
    }
    else {
      // other attributes
      assert(false && "Not Implemented yet");
    }
    optionalAdvance(tk_Comma);
  }
  advance(tk_R_SquareBracket);
  return;
}

std::unique_ptr<ast::CircuitStmt> Parser::parseCircuitStmt() {
  advance(tk_Circuit);

  auto stmt = std::make_unique<ast::CircuitStmt>();
  parseAttribute(stmt->attribute);
  requireCurTokenIs(tk_Identifier);
  stmt->name = curToken.toString();
  advance(tk_Identifier);

  advance(tk_L_CurlyBracket);
  // TODO: parse circuit body

  advance(tk_R_CurlyBracket);

  return stmt;
}

ast::RootNode Parser::parse() {
  ast::RootNode root;
  while (curToken.isNot(tk_Eof)) {
    switch (curToken.kind) {
      case tk_Circuit: {
        root.stmts.emplace_back(std::move(parseCircuitStmt()));
        break;
      }
      default: {
        logErr() << "Unknown statement. CurToken is ";
        curToken.print() << "\n";
        printLocation();
        failAndExit();
      }
    }
  }
  return root;
}