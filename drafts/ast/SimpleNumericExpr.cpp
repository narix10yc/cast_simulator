#include "new_parser/AST.h"
#include "new_parser/ASTContext.h"
#include "llvm/Support/Casting.h"

#include <tuple>
#include <numeric> // for std::gcd

using namespace cast::draft::ast;

namespace {
  struct Fraction {
    int n;
    int d;
  };

  Fraction simplifyFraction(int n, int d) {
    assert(d != 0 && "Denominator must not be zero");
    if (n == 0)
      return {0, 1};
    if (d < 0) {
      n = -n;
      d = -d;
    }
    auto g = std::gcd(n, d);
    return {n, d};
  }
}


FractionLiteral::FractionLiteral(int numerator, int denominator)
    : SimpleNumericExpr(NK_Expr_FractionLiteral) {
  auto [n, d] = simplifyFraction(numerator, denominator);
  this->n = n;
  this->d = d;
}

FractionPiLiteral::FractionPiLiteral(int numerator, int denominator)
    : SimpleNumericExpr(NK_Expr_FractionPiLiteral) {
  auto [n, d] = simplifyFraction(numerator, denominator);
  this->n = n;
  this->d = d;
}

/* Print */

std::ostream& FractionLiteral::print(std::ostream& os) const {
  assert(d > 0 && "Denominator must be positive");
  if (d == 1)
    return os << n;
  return os << n << "/" << d;
}

std::ostream& FractionPiLiteral::print(std::ostream& os) const {
  assert(d > 0 && "Denominator must be positive");
  if (n == 1) {}
  else if (n == -1) { os << "-"; }
  else { os << n << "*"; }
  
  os << "Pi";

  if (d != 1)
    os << "/" << d;

  return os;
}

/* Arithmatics */

SimpleNumericExpr* SimpleNumericExpr::neg(
    ASTContext& ctx, SimpleNumericExpr* operand) {
  assert(operand && "Operand cannot be null");
  if (auto* e = llvm::dyn_cast<IntegerLiteral>(operand))
    return new (ctx) IntegerLiteral(-e->value);
  if (auto* e = llvm::dyn_cast<FloatingLiteral>(operand))
    return new (ctx) FloatingLiteral(-e->value);
  if (auto* e = llvm::dyn_cast<FractionLiteral>(operand))
    return new (ctx) FractionLiteral(-e->n, e->d);
  if (auto* e = llvm::dyn_cast<FractionPiLiteral>(operand))
    return new (ctx) FractionPiLiteral(-e->n, e->d);
  assert(false && "Invalid state");
  return nullptr;
}

SimpleNumericExpr* SimpleNumericExpr::add(
    ASTContext& ctx, SimpleNumericExpr* lhs, SimpleNumericExpr* rhs) {
  assert(lhs && rhs && "Operands cannot be null");
  if (auto* L = llvm::dyn_cast<IntegerLiteral>(lhs)) {
    // okay: int + int
    if (auto* R = llvm::dyn_cast<IntegerLiteral>(rhs))
      return new (ctx) IntegerLiteral(L->value + R->value);
    // okay: int + fraction
    if (auto* R = llvm::dyn_cast<FractionLiteral>(rhs))
      return new (ctx) FractionLiteral(
          L->value * R->d + R->n,
          R->d);
  }
  if (auto* L = llvm::dyn_cast<FractionLiteral>(lhs)) {
    // okay: fraction + int
    if (auto* R = llvm::dyn_cast<IntegerLiteral>(rhs))
      return new (ctx) FractionLiteral(
          L->n + R->value * L->d,
          L->d);
    // okay: fraction + fraction
    if (auto* R = llvm::dyn_cast<FractionLiteral>(rhs))
      return new (ctx) FractionLiteral(
          L->n * R->d + R->n * L->d,
          L->d * R->d);
  }

  // okay: fractionPi + fractionPi
  if (auto* L = llvm::dyn_cast<FractionPiLiteral>(lhs)) {
    if (auto* R = llvm::dyn_cast<FractionPiLiteral>(rhs))
      return new (ctx) FractionPiLiteral(
          L->n * R->d + R->n * L->d,
          L->d * R->d);
  }

  // otherwise, use floating point
  return new (ctx) FloatingLiteral(lhs->getValue() + rhs->getValue());
}

SimpleNumericExpr* SimpleNumericExpr::sub(
    ASTContext& ctx, SimpleNumericExpr* lhs, SimpleNumericExpr* rhs) {
  assert(lhs && rhs && "Operands cannot be null");
  if (auto* L = llvm::dyn_cast<IntegerLiteral>(lhs)) {
    // okay: int - int
    if (auto* R = llvm::dyn_cast<IntegerLiteral>(rhs))
      return new (ctx) IntegerLiteral(L->value - R->value);
    // okay: int - fraction
    if (auto* R = llvm::dyn_cast<FractionLiteral>(rhs))
      return new (ctx) FractionLiteral(
          L->value * R->d - R->n,
          R->d);
  }
  if (auto* L = llvm::dyn_cast<FractionLiteral>(lhs)) {
    // okay: fraction - int
    if (auto* R = llvm::dyn_cast<IntegerLiteral>(rhs))
      return new (ctx) FractionLiteral(
          L->n - R->value * L->d,
          L->d);
    // okay: fraction - fraction
    if (auto* R = llvm::dyn_cast<FractionLiteral>(rhs))
      return new (ctx) FractionLiteral(
          L->n * R->d - R->n * L->d,
          L->d * R->d);
  }

  // okay: fractionPi - fractionPi
  if (auto* L = llvm::dyn_cast<FractionPiLiteral>(lhs)) {
    if (auto* R = llvm::dyn_cast<FractionPiLiteral>(rhs))
      return new (ctx) FractionPiLiteral(
          L->n * R->d - R->n * L->d,
          L->d * R->d);
  }

  // otherwise, use floating point
  return new (ctx) FloatingLiteral(lhs->getValue() - rhs->getValue());
}

SimpleNumericExpr* SimpleNumericExpr::mul(
    ASTContext& ctx, SimpleNumericExpr* lhs, SimpleNumericExpr* rhs) {
  assert(lhs && rhs && "Operands cannot be null");
  if (auto* L = llvm::dyn_cast<IntegerLiteral>(lhs)) {
    // okay: int * int
    if (auto* R = llvm::dyn_cast<IntegerLiteral>(rhs))
      return new (ctx) IntegerLiteral(L->value * R->value);
    // okay: int * fraction
    if (auto* R = llvm::dyn_cast<FractionLiteral>(rhs))
      return new (ctx) FractionLiteral(
          L->value * R->n,
          R->d);
    // okay: int * fractionPi
    if (auto* R = llvm::dyn_cast<FractionPiLiteral>(rhs))
      return new (ctx) FractionPiLiteral(
          L->value * R->n,
          R->d);
  }
  if (auto* L = llvm::dyn_cast<FractionLiteral>(lhs)) {
    // okay: fraction * int
    if (auto* R = llvm::dyn_cast<IntegerLiteral>(rhs))
      return new (ctx) FractionLiteral(
          L->n * R->value,
          L->d);
    // okay: fraction * fraction
    if (auto* R = llvm::dyn_cast<FractionLiteral>(rhs))
      return new (ctx) FractionLiteral(
          L->n * R->n,
          L->d * R->d);
    // okay: fraction * fractionPi
    if (auto* R = llvm::dyn_cast<FractionPiLiteral>(rhs))
      return new (ctx) FractionPiLiteral(
          L->n * R->n,
          L->d * R->d);
  }

  if (auto* L = llvm::dyn_cast<FractionPiLiteral>(lhs)) {
    // okay: fractionPi * int
    if (auto* R = llvm::dyn_cast<IntegerLiteral>(rhs))
      return new (ctx) FractionPiLiteral(
          L->n * R->value,
          L->d);
    // okay: fractionPi * fraction
    if (auto* R = llvm::dyn_cast<FractionLiteral>(rhs))
      return new (ctx) FractionPiLiteral(
          L->n * R->n,
          L->d * R->d);
  }

  // otherwise, use floating point
  return new (ctx) FloatingLiteral(lhs->getValue() * rhs->getValue());
}

SimpleNumericExpr* SimpleNumericExpr::div(
    ASTContext& ctx, SimpleNumericExpr* lhs, SimpleNumericExpr* rhs) {
  assert(lhs && rhs && "Operands cannot be null");
  if (auto* L = llvm::dyn_cast<IntegerLiteral>(lhs)) {
    // okay: int / int
    if (auto* R = llvm::dyn_cast<IntegerLiteral>(rhs))
      return new (ctx) FractionLiteral(L->value, R->value);
    // okay: int / fraction
    if (auto* R = llvm::dyn_cast<FractionLiteral>(rhs))
      return new (ctx) FractionLiteral(
          L->value * R->d,
          R->n);
  }
  if (auto* L = llvm::dyn_cast<FractionLiteral>(lhs)) {
    // okay: fraction / int
    if (auto* R = llvm::dyn_cast<IntegerLiteral>(rhs))
      return new (ctx) FractionLiteral(
          L->n,
          L->d * R->value);
    // okay: fraction / fraction
    if (auto* R = llvm::dyn_cast<FractionLiteral>(rhs))
      return new (ctx) FractionLiteral(
          L->n * R->d,
          L->d * R->n);
  }

  // okay: fractionPi / int
  if (auto* L = llvm::dyn_cast<FractionPiLiteral>(lhs)) {
    if (auto* R = llvm::dyn_cast<IntegerLiteral>(rhs))
      return new (ctx) FractionPiLiteral(
          L->n,
          L->d * R->value);
  }

  // otherwise, use floating point
  return new (ctx) FloatingLiteral(lhs->getValue() / rhs->getValue());
}