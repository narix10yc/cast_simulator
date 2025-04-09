#include "new_parser/Parser.h"

using namespace cast::draft;

std::ostream& ast::BinaryOpExpr::print(std::ostream& os) const {
  os << "(";
  lhs->print(os);
  os << " ";
  switch (op) {
    case Add: os << "+"; break;
    case Sub: os << "-"; break;
    case Mul: os << "*"; break;
    case Div: os << "/"; break;
    case Pow: os << "**"; break;
    default: 
      assert(false && "Invalid operator");
  }
  rhs->print(os << " ");
  return os << ")";
}

double ast::SimpleNumericExpr::getValue() const {
  if (_value.is<int>())
    return static_cast<double>(_value.get<int>());
  if (_value.is<double>())
    return _value.get<double>();
  assert(_value.is<FractionPi>() && "Invalid state");
  const auto& fraction = _value.get<FractionPi>();
  return static_cast<double>(fraction.numerator) / fraction.denominator * M_PI;
}

/* Arithmatic of SimpleNumericExpr */
ast::SimpleNumericExpr
ast::SimpleNumericExpr::operator+(const SimpleNumericExpr& other) const {
  if (_value.is<double>() || other._value.is<double>())
    return SimpleNumericExpr(getValue() + other.getValue());

  // okay: int + int
  if (_value.is<int>() && other._value.is<int>())
    return SimpleNumericExpr(_value.get<int>() + other._value.get<int>());
  
  // okay: FractionPi + FractionPi
  if (_value.is<FractionPi>() && other._value.is<FractionPi>()) {
    const auto& lhs = _value.get<FractionPi>();
    const auto& rhs = other._value.get<FractionPi>();
    return SimpleNumericExpr(
      lhs.numerator * rhs.denominator + rhs.numerator * lhs.denominator,
      lhs.denominator * rhs.denominator);
  }
  
  // not okay otherwise
  SimpleNumericExpr reduced(getValue() + other.getValue()); 
  #ifndef NDEBUG
    std::cerr << BOLDYELLOW("Warning: ") << "Expression ";
    print(std::cerr) << " + ";
    other.print(std::cerr) << " is reduced to ";
    reduced.print(std::cerr) << "\n";
  #endif
  return reduced;
}

ast::SimpleNumericExpr
ast::SimpleNumericExpr::operator-(const SimpleNumericExpr& other) const {
  if (_value.is<double>() || other._value.is<double>())
    return SimpleNumericExpr(getValue() - other.getValue());

  // okay: int - int
  if (_value.is<int>() && other._value.is<int>())
    return SimpleNumericExpr(_value.get<int>() - other._value.get<int>());

  // okay: FractionPi - FractionPi
  if (_value.is<FractionPi>() && other._value.is<FractionPi>()) {
    const auto& lhs = _value.get<FractionPi>();
    const auto& rhs = other._value.get<FractionPi>();
    return SimpleNumericExpr(
      lhs.numerator * rhs.denominator - rhs.numerator * lhs.denominator,
      lhs.denominator * rhs.denominator);
  }

  // not okay otherwise
  SimpleNumericExpr reduced(getValue() - other.getValue()); 
  #ifndef NDEBUG
    std::cerr << BOLDYELLOW("Warning: ") << "Expression ";
    print(std::cerr) << " - ";
    other.print(std::cerr) << " is reduced to ";
    reduced.print(std::cerr) << "\n";
  #endif
  return reduced;
}

ast::SimpleNumericExpr
ast::SimpleNumericExpr::operator*(const SimpleNumericExpr& other) const {
  if (_value.is<double>() || other._value.is<double>())
    return SimpleNumericExpr(getValue() * other.getValue());

  // okay: int * int
  if (_value.is<int>() && other._value.is<int>())
    return SimpleNumericExpr(_value.get<int>() * other._value.get<int>());

  // okay: FractionPi * int
  if (_value.is<FractionPi>() && other._value.is<int>()) {
    const auto& lhs = _value.get<FractionPi>();
    return SimpleNumericExpr(
      lhs.numerator * other._value.get<int>(),
      lhs.denominator);
  }

  // okay: int * FractionPi
  if (_value.is<int>() && other._value.is<FractionPi>()) {
    const auto& rhs = other._value.get<FractionPi>();
    return SimpleNumericExpr(
      _value.get<int>() * rhs.numerator,
      rhs.denominator);
  }

  // not okay otherwise
  SimpleNumericExpr reduced(getValue() * other.getValue()); 
  #ifndef NDEBUG
    std::cerr << BOLDYELLOW("Warning: ") << "Expression ";
    print(std::cerr) << " * ";
    other.print(std::cerr) << " is reduced to ";
    reduced.print(std::cerr) << "\n";
  #endif
  return reduced;

}

ast::SimpleNumericExpr
ast::SimpleNumericExpr::operator/(const SimpleNumericExpr& other) const {
  if (_value.is<double>() || other._value.is<double>())
    return SimpleNumericExpr(getValue() / other.getValue());
  
  // okay: int / int
  if (_value.is<int>() && other._value.is<int>()) {
    #ifndef NDEBUG
      std::cerr << BOLDYELLOW("Known Issue Warning: ")
                << "Fraction information ";
      print(std::cerr) << " / ";
      other.print(std::cerr) << " is lost.\n";
    #endif
    return SimpleNumericExpr(getValue() / other.getValue());
  }
  
  // okay: FractionPi / int
  if (_value.is<FractionPi>() && other._value.is<int>()) {
    const auto& lhs = _value.get<FractionPi>();
    return SimpleNumericExpr(
      lhs.numerator,
      lhs.denominator * other._value.get<int>());
  }

  // not okay otherwise
  SimpleNumericExpr reduced(getValue() / other.getValue()); 
  #ifndef NDEBUG
    std::cerr << BOLDYELLOW("Warning: ") << "Expression ";
    print(std::cerr) << " / ";
    other.print(std::cerr) << " is reduced to ";
    reduced.print(std::cerr) << "\n";
  #endif
  return reduced;
}

static int getPrecedence(ast::BinaryOpExpr::BinaryOpKind binOp) {
  switch (binOp) {
  case ast::BinaryOpExpr::Invalid:
    return -1;
  case ast::BinaryOpExpr::Add:
  case ast::BinaryOpExpr::Sub:
    return 10;
  case ast::BinaryOpExpr::Mul:
  case ast::BinaryOpExpr::Div:
    return 20;
  case ast::BinaryOpExpr::Pow:
    return 30;
  default:
    assert(false && "Invalid binary operator");
    return -1;
  }
}

static ast::BinaryOpExpr::BinaryOpKind toBinaryOp(TokenKind tokenKind) {
  switch (tokenKind) {
  case tk_Add:
    return ast::BinaryOpExpr::Add;
  case tk_Sub:
    return ast::BinaryOpExpr::Sub;
  case tk_Mul:
    return ast::BinaryOpExpr::Mul;
  case tk_Div:
    return ast::BinaryOpExpr::Div;
  case tk_Pow:
    return ast::BinaryOpExpr::Pow;
  default:
    return ast::BinaryOpExpr::Invalid;
  }
}

std::unique_ptr<ast::Expr> Parser::parsePrimaryExpr() {
  if (curToken.is(tk_Numeric)) {
    if (curToken.convertibleToInt()) {
      auto value = curToken.toInt();
      advance(tk_Numeric);
      return std::make_unique<ast::SimpleNumericExpr>(value);
    }
    auto value = curToken.toDouble();
    advance(tk_Numeric);
    return std::make_unique<ast::SimpleNumericExpr>(value);
  }
  if (curToken.is(tk_Pi)) {
    advance(tk_Pi);
    return std::make_unique<ast::SimpleNumericExpr>(1, 1);
  }
  if (curToken.is(tk_Hash)) {
    advance(tk_Hash);
    requireCurTokenIs(tk_Numeric, "Expect a number after #");
    auto index = curToken.toInt();
    advance(tk_Numeric);
    return std::make_unique<ast::ParameterExpr>(index);
  }
  if (curToken.is(tk_L_RoundBracket)) {
    advance(tk_L_RoundBracket);
    auto expr = parseExpr();
    requireCurTokenIs(tk_R_RoundBracket, "Expect ')' to close the expression");
    advance(tk_R_RoundBracket);
    return expr;
  }
  logErr() << "Unknown expression. CurToken is ";
  curToken.print(std::cerr) << "\n";
  failAndExit();
  return nullptr;
}

std::unique_ptr<ast::Expr> Parser::parseExpr(int precedence) {
  auto lhs = parsePrimaryExpr();
  while (true) {
    auto binOp = toBinaryOp(curToken.kind);
    int prec = getPrecedence(binOp);
    if (prec < precedence)
      break;
    advance();
    auto rhs = parseExpr(prec + 1);
    lhs = std::make_unique<ast::BinaryOpExpr>(
      binOp, std::move(lhs), std::move(rhs));
  }
  return lhs;
}