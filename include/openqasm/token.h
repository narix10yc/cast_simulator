#ifndef OPENQASM_TOKEN_H
#define OPENQASM_TOKEN_H

#include <string>

namespace openqasm {

enum class TokenTy;
enum class UnaryOp;
enum class BinaryOp;

enum class TokenTy : int {
  Eof = -1,
  Identifier = -2,
  Numeric = -3,

  // keywords
  If = -10,
  Then = -11,
  Else = -12,
  Openqasm = -13,
  Include = -14,
  Qreg = -15,
  Creg = -16,
  Gate = -17,

  // operators
  Add = -30,
  Sub = -31,
  Mul = -32,
  Div = -33,
  Greater = -34,
  Less = -35,
  GreaterEqual = -36,
  LessEqual = -37,

  Comma = -104,          // ,
  Semicolon = -105,      // ;
  L_RoundBracket = -106,  // (
  R_RoundBracket = -107,  // )
  L_SquareBracket = -108, // [
  R_SquareBracket = -109, // ]
  L_AngleBracket = -110,  // <
  R_AngleBracket = -111,  // >
  L_CurlyBracket = -112,  // {
  R_CurlyBracket = -113,  // }
  SingleQuote = -114,    // '
  DoubleQuote = -115,    // "

  LineFeed = 10,       // '\n'
  CarriageReturn = 13, // '\r'

  Unknown = -1000,
};

enum class UnaryOp {
  Positive,
  Negative,
  None,
};

enum class BinaryOp {
  Greater,
  Less,
  GreaterEqual,
  LessEqual,
  Add,
  Sub,
  Mul,
  Div,
  None,
};

class Token {
public:
  TokenTy type;
  std::string str;
  Token(TokenTy type) : type(type), str("") {}
  Token(TokenTy type, const std::string& str) : type(type), str(str) {}

  std::string toString() const;

  BinaryOp toBinaryOp() const {
    switch (type) {
    case TokenTy::Greater:
      return BinaryOp::Greater;
    case TokenTy::Less:
      return BinaryOp::Less;
    case TokenTy::GreaterEqual:
      return BinaryOp::GreaterEqual;
    case TokenTy::LessEqual:
      return BinaryOp::LessEqual;
    case TokenTy::Add:
      return BinaryOp::Add;
    case TokenTy::Sub:
      return BinaryOp::Sub;
    case TokenTy::Mul:
      return BinaryOp::Mul;
    case TokenTy::Div:
      return BinaryOp::Div;
    default:
      return BinaryOp::None;
    }
  }
};

std::string tokenTypetoString(TokenTy ty);

int getBinopPrecedence(BinaryOp op);

} // namespace openqasm

#endif // OPENQASM_TOKEN_H