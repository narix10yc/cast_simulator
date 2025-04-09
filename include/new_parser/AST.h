#ifndef CAST_DRAFT_AST_H
#define CAST_DRAFT_AST_H

#include "llvm/Support/Casting.h"
#include "utils/PODVariant.h"
#include <iostream>
#include <string>

namespace cast::draft {

namespace ast {

class Node {
public:
  /// The LLVM-style RTTI. Indentation corresponds to class hirearchy.
  enum NodeKind {
    NK_Stmt,
      NK_Stmt_Circuit,
      NK_Stmt_GateApply,
    NK_Expr,
      NK_Expr_SimpleNumeric,
      NK_Expr_Parameter,
      NK_Expr_BinaryOp,
      NK_Expr_MinusOp,
    NK_Root
  };
private:
  NodeKind _kind;
public:
  explicit Node(NodeKind kind) : _kind(kind) {}

  NodeKind getKind() const { return _kind; }

  virtual ~Node() = default;

  virtual std::ostream& print(std::ostream& os) const = 0;
}; // class Node

class Expr : public Node {
public:
  explicit Expr(NodeKind kind) : Node(kind) {}

  static bool classof(const Node* node) {
    return node->getKind() >= NK_Expr && node->getKind() <= NK_Expr_MinusOp;
  }
}; // class Expr

/// @brief SimpleNumericExpr represents a constant numeric expression that can
/// be either a double or a fraction of pi. This design allows storing exact
/// value such as pi/2 and 2*pi/3 that are useful in printing the AST.
class SimpleNumericExpr : public Expr {
private:
  struct FractionPi {
    int numerator;
    int denominator;
  };
  utils::PODVariant<int, double, FractionPi> _value;
public:
  SimpleNumericExpr() : Expr(NK_Expr_SimpleNumeric), _value() {}

  explicit SimpleNumericExpr(int value)
    : Expr(NK_Expr_SimpleNumeric)
    , _value(value) {}

  explicit SimpleNumericExpr(double value)
    : Expr(NK_Expr_SimpleNumeric)
    , _value(value) {}
  
  explicit SimpleNumericExpr(int numerator, int denominator)
    : Expr(NK_Expr_SimpleNumeric)
    , _value(FractionPi{numerator, denominator}) {}

  std::ostream& print(std::ostream& os) const override;

  double getValue() const;

  SimpleNumericExpr operator-() const;

  SimpleNumericExpr operator+(const SimpleNumericExpr& other) const;
  SimpleNumericExpr operator-(const SimpleNumericExpr& other) const;
  SimpleNumericExpr operator*(const SimpleNumericExpr& other) const;
  // TODO: Known issue: fraction information in non-multiple-of-pi will be lost
  // For example, 2*pi/3 is correctly handled, but 2/3*pi is not.
  SimpleNumericExpr operator/(const SimpleNumericExpr& other) const;

  static bool classof(const Node* node) {
    return node->getKind() == NK_Expr_SimpleNumeric;
  }
}; // class SimpleNumericExpr

// #index 
class ParameterExpr : public Expr {
public:
  int index;
  ParameterExpr(int index)
    : Expr(NK_Expr_Parameter), index(index) {}

  std::ostream& print(std::ostream& os) const override {
    return os << "#" << index;
  }

  static bool classof(const Node* node) {
    return node->getKind() == NK_Expr_Parameter;
  }
}; // class ParameterExpr

class BinaryOpExpr : public Expr {
public:
  enum BinaryOpKind {
    Invalid,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
  };

  BinaryOpKind op;
  std::unique_ptr<Expr> lhs;
  std::unique_ptr<Expr> rhs;

  BinaryOpExpr(
    BinaryOpKind op, std::unique_ptr<Expr> lhs, std::unique_ptr<Expr> rhs)
  : Expr(NK_Expr_BinaryOp), op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  std::ostream& print(std::ostream& os) const override;

  static bool classof(const Node* node) {
    return node->getKind() == NK_Expr_BinaryOp;
  }

}; // class BinaryOpExpr

class MinusOpExpr : public Expr {
public:
  std::unique_ptr<Expr> operand;

  MinusOpExpr(std::unique_ptr<Expr> operand)
    : Expr(NK_Expr_MinusOp), operand(std::move(operand)) {}

  std::ostream& print(std::ostream& os) const override;

  static bool classof(const Node* node) {
    return node->getKind() == NK_Expr_MinusOp;
  }
}; // class UnaryOpExpr

class Attribute {
public:
  int nQubits;
  int nParams;
  SimpleNumericExpr phase;

  std::ostream& print(std::ostream& os) const;
};

class Stmt : public Node {
public:
  Attribute attribute;

  explicit Stmt(NodeKind kind) : Node(kind), attribute() {}

  void setNQubits(int nQubits) { attribute.nQubits = nQubits; }

  void setNParams(int nParams) { attribute.nParams = nParams; }

  void setPhase(const SimpleNumericExpr& phase) { attribute.phase = phase; }

  static bool classof(const Node* node) {
    return node->getKind() >= NK_Stmt && node->getKind() <= NK_Stmt_GateApply;
  }
}; // class Stmt

class CircuitStmt : public Stmt {
public:
  std::string name;

  CircuitStmt() : Stmt(NK_Stmt_Circuit), name() {}
  CircuitStmt(const std::string& name) : Stmt(NK_Stmt_Circuit), name(name) {}

  std::ostream& print(std::ostream& os) const override;

  static bool classof(const Node* node) {
    return node->getKind() == NK_Stmt_Circuit;
  }
}; // class CircuitStmt

class RootNode : public Node {
public:
  RootNode() : Node(NK_Root), stmts() {}
  
  std::vector<std::unique_ptr<Stmt>> stmts;

  std::ostream& print(std::ostream& os) const override;

  static bool classof(const Node* node) {
    return node->getKind() == NK_Root;
  }
}; // class RootNode

} // namespace ast
} // namespace cast::draft

#endif // CAST_DRAFT_AST_H