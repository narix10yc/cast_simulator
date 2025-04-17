#ifndef CAST_DRAFT_AST_H
#define CAST_DRAFT_AST_H

#include "llvm/Support/Casting.h"
#include "llvm/ADT/SmallVector.h"
#include "utils/PODVariant.h"
#include <iostream>
#include <string>
#include <span>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cast {
  class CircuitGraph;
} // namespace cast

namespace cast::draft {
  class ASTContext;

namespace ast {

class Node {
public:
  /// The LLVM-style RTTI. Indentation corresponds to class hirearchy.
  enum NodeKind {
    NK_Stmt,
      NK_Stmt_GateApply,
      NK_Stmt_GateChain,
      NK_Stmt_GateBlock,
      NK_Stmt_Measure,
      NK_Stmt_If,
      NK_Stmt_Repeat,
      NK_Stmt_Circuit,
      NK_Stmt_Channel,
    NK_Expr,
      NK_Expr_SimpleNumeric,
        NK_Expr_IntegerLiteral,
        NK_Expr_FloatingLiteral,
        NK_Expr_FractionLiteral,
        NK_Expr_FractionPiLiteral,
      NK_Expr_Measure,
      NK_Expr_All,
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

/// @brief SimpleNumericExpr is a purely abstract class that represents a
/// constant numeric expression. This
/// design allows storing exact values of fraction-multiple of pi, such as pi/2
/// and 2*pi/3. We also support basic arithmatics. They are useful in printing
/// the AST.
class SimpleNumericExpr : public Expr {
public:
  explicit SimpleNumericExpr(NodeKind kind) : Expr(kind) {
    assert(llvm::dyn_cast<SimpleNumericExpr>(this) != nullptr);
  }
 
  virtual double getValue() const = 0;

  // "-" <SimpleNumericExpr>
  static SimpleNumericExpr* neg(ASTContext& ctx, SimpleNumericExpr* operand);

  // <SimpleNumericExpr> "+" <SimpleNumericExpr>
  static SimpleNumericExpr*
  add(ASTContext& ctx, SimpleNumericExpr* lhs, SimpleNumericExpr* rhs);

  // <SimpleNumericExpr> "-" <SimpleNumericExpr>
  static SimpleNumericExpr*
  sub(ASTContext& ctx, SimpleNumericExpr* lhs, SimpleNumericExpr* rhs);
  
  // <SimpleNumericExpr> "*" <SimpleNumericExpr>
  static SimpleNumericExpr*
  mul(ASTContext& ctx, SimpleNumericExpr* lhs, SimpleNumericExpr* rhs);

  // <SimpleNumericExpr> "/" <SimpleNumericExpr>
  static SimpleNumericExpr*
  div(ASTContext& ctx, SimpleNumericExpr* lhs, SimpleNumericExpr* rhs);

  static bool classof(const Node* node) {
    return node->getKind() >= NK_Expr_SimpleNumeric &&
           node->getKind() <= NK_Expr_FractionPiLiteral;
  }
}; // class SimpleNumericExpr

class IntegerLiteral : public SimpleNumericExpr {
public:
  int value;

  IntegerLiteral(int value)
    : SimpleNumericExpr(NK_Expr_IntegerLiteral), value(value) {}

  double getValue() const override { return value; }

  std::ostream& print(std::ostream& os) const override {
    return os << value;
  }

  static bool classof(const Node* node) {
    return node->getKind() == NK_Expr_IntegerLiteral;
  }
}; // class IntegerLiteral

class FloatingLiteral : public SimpleNumericExpr {
public:
  double value;

  FloatingLiteral(double value)
    : SimpleNumericExpr(NK_Expr_FloatingLiteral), value(value) {}

  double getValue() const override { return value; }

  std::ostream& print(std::ostream& os) const override {
    return os << value;
  }

  static bool classof(const Node* node) {
    return node->getKind() == NK_Expr_FloatingLiteral;
  }
}; // class FloatingLiteral

class FractionLiteral : public SimpleNumericExpr {
public:
  // numerator
  int n;
  // denominator (always positive)
  int d;

  // Handle fraction simplification
  FractionLiteral(int numerator, int denominator);

  double getValue() const override { return double(n) / d; }

  std::ostream& print(std::ostream& os) const override;

  static bool classof(const Node* node) {
    return node->getKind() == NK_Expr_FractionLiteral;
  }
}; // class FractionLiteral

class FractionPiLiteral : public SimpleNumericExpr {
public:
  // numerator
  int n;
  // denominator (always positive)
  int d;

  FractionPiLiteral(int numerator, int denominator);

  double getValue() const override { 
    return M_PI * n / d;
  }

  std::ostream& print(std::ostream& os) const override;
  
  static bool classof(const Node* node) {
    return node->getKind() == NK_Expr_FractionPiLiteral;
  }
}; // class FractionPiLiteral

class MeasureExpr : public Expr {
public:
  Expr* target;

  MeasureExpr(Expr* target)
    : Expr(NK_Expr_Measure), target(target) {}

  std::ostream& print(std::ostream& os) const override {
    return target->print(os << "Measure ");
  }

  static bool classof(const Node* node) {
    return node->getKind() == NK_Expr_Measure;
  }
}; // class MeasureExpr

// \c AllExpr corresponds to the 'All' keyword that is used as an convenient way
// of applying a gate to all wires.
class AllExpr : public Expr {
public:
  AllExpr() : Expr(NK_Expr_All) {}

  std::ostream& print(std::ostream& os) const override {
    return os << "All";
  }

  static bool classof(const Node* node) {
    return node->getKind() == NK_Expr_All;
  }

}; // class AllExpr

// "#" <uint>
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
  Expr* lhs;
  Expr* rhs;

  BinaryOpExpr(BinaryOpKind op, Expr* lhs, Expr* rhs)
    : Expr(NK_Expr_BinaryOp), op(op), lhs(lhs), rhs(rhs) {}

  std::ostream& print(std::ostream& os) const override;

  static bool classof(const Node* node) {
    return node->getKind() == NK_Expr_BinaryOp;
  }

}; // class BinaryOpExpr

// "-" <expr>
class MinusOpExpr : public Expr {
public:
  Expr* operand;

  MinusOpExpr(Expr* operand)
    : Expr(NK_Expr_MinusOp), operand(operand) {}

  std::ostream& print(std::ostream& os) const override;

  static bool classof(const Node* node) {
    return node->getKind() == NK_Expr_MinusOp;
  }
}; // class UnaryOpExpr

class Attribute {
public:
  // we use pointer here to allow null
  IntegerLiteral* nQubits;
  IntegerLiteral* nParams;
  SimpleNumericExpr* phase;
  
  Attribute(IntegerLiteral* nQubits,
            IntegerLiteral* nParams,
            SimpleNumericExpr* phase)
    : nQubits(nQubits), nParams(nParams), phase(phase) {}

  std::ostream& print(std::ostream& os) const;
};

class Stmt : public Node {
public:
  explicit Stmt(NodeKind kind) : Node(kind) {}

  static bool classof(const Node* node) {
    return node->getKind() >= NK_Stmt && node->getKind() <= NK_Stmt_Channel;
  }
}; // class Stmt

class GateApplyStmt : public Stmt {
public:
  std::string_view name;
  std::span<Expr*> params;
  std::span<Expr*> qubits;

  GateApplyStmt(std::string_view name,
                std::span<Expr*> params,
                std::span<Expr*> qubits)
    : Stmt(NK_Stmt_GateApply), name(name), params(params), qubits(qubits) {}

  // Because we expect \c GateApplyStmt will not appear in the top-level, 
  // \c print does not print indentation or the final semicolon.
  std::ostream& print(std::ostream& os) const override;

  static bool classof(const Node* node) {
    return node->getKind() == NK_Stmt_GateApply;
  }
}; // class GateApplyStmt

class GateChainStmt : public Stmt {
public:
  std::span<GateApplyStmt*> gates;

  GateChainStmt(std::span<GateApplyStmt*> gates)
    : Stmt(NK_Stmt_GateChain), gates(gates) {}

  std::ostream& print(std::ostream& os) const override;

  static bool classof(const Node* node) {
    return node->getKind() == NK_Stmt_GateChain;
  }
}; // class GateChainStmt

class MeasureStmt : public Stmt {
public:
  Expr* target;

  MeasureStmt(Expr* target)
    : Stmt(NK_Stmt_Measure), target(target) {}

  std::ostream& print(std::ostream& os) const override {
    return target->print(os << "Measure ");
  }

  static bool classof(const Node* node) {
    return node->getKind() == NK_Stmt_Measure;
  }
}; // class MeasureStmt

// "Circuit" ["<" <attribute> ">"] <name> "{" {<stmt>} "}"
class CircuitStmt : public Stmt {
public:
  std::string_view name;
  Attribute* attr;
  std::span<Stmt*> body;

  CircuitStmt(std::string_view name)
    : Stmt(NK_Stmt_Circuit), name(name), attr(nullptr), body() {}

  CircuitStmt(std::string_view name, Attribute* attr, std::span<Stmt*> body)
    : Stmt(NK_Stmt_Circuit), name(name), attr(attr), body(body) {}

  std::ostream& print(std::ostream& os) const override;

  void toCircuitGraph(cast::CircuitGraph& graph) const;

  static bool classof(const Node* node) {
    return node->getKind() == NK_Stmt_Circuit;
  }
}; // class CircuitStmt

class RootNode : public Node {
public:
  std::span<Stmt*> stmts;

  RootNode(std::span<Stmt*> stmts) : Node(NK_Root), stmts(stmts) {}

  std::ostream& print(std::ostream& os) const override;

  /// Lookup a circuit by name. If not found, return nullptr.
  /// If name is empty, return the first circuit found.
  CircuitStmt* lookupCircuit(const std::string& name = "");

  static bool classof(const Node* node) {
    return node->getKind() == NK_Root;
  }
}; // class RootNode

} // namespace ast
} // namespace cast::draft

#endif // CAST_DRAFT_AST_H