#ifndef OPENQASM_AST_H
#define OPENQASM_AST_H

#include "openqasm/token.h"
#include "utils/utils.h"

#include <fstream>
#include <memory>

namespace cast {
  class LegacyCircuitGraph;
}
namespace openqasm::ast {

class Node;
class RootNode;

class Statement;

class Expression;
class NumericExpr;
class VariableExpr;
class UnaryExpr;
class BinaryExpr;

class ExpressionValue {
public:
  const bool isConstant;
  const double value;
  const bool isMultipleOfPI;
  ExpressionValue(double value)
      : isConstant(true), value(value), isMultipleOfPI(false) {}
  ExpressionValue(bool isConstant)
      : isConstant(isConstant), value(0), isMultipleOfPI(false) {}
  ExpressionValue(bool isConstant, double value, bool isMultipleOfPI)
      : isConstant(isConstant), value(value), isMultipleOfPI(isMultipleOfPI) {}

  static ExpressionValue MultipleOfPI(double m) { return {true, m, true}; }
};

class Node {
public:
  virtual ~Node() = default;
  virtual std::string toString() const = 0;
  virtual void prettyPrint(std::ostream& f, int depth) const = 0;
};

class Statement : public Node {
public:
  std::string toString() const override { return "statement"; }
  void prettyPrint(std::ostream& f, int depth) const override {}
};

class Expression : public Node {
public:
  std::string toString() const override { return "expression"; }
  void prettyPrint(std::ostream& f, int depth) const override {}
  virtual ExpressionValue getExprValue() const { return false; }
};

class NumericExpr : public Expression {
  double value;

public:
  NumericExpr(double value) : value(value) {}
  std::string toString() const override {
    return "(" + std::to_string(value) + ")";
  }

  void prettyPrint(std::ostream& f, int depth) const override;

  double getValue() const { return value; }
  ExpressionValue getExprValue() const override { return value; }
};

class VariableExpr : public Expression {
  std::string name;

public:
  VariableExpr(std::string name) : name(name) {}

  std::string getName() const { return name; }

  std::string toString() const override { return "(" + name + ")"; }
  void prettyPrint(std::ostream& f, int depth) const override;

  ExpressionValue getExprValue() const override {
    return (name == "pi") ? 3.14159265358979323846 : false;
  }
};

class SubscriptExpr : public Expression {
public:
  std::string name;
  int index;
  SubscriptExpr(std::string name, int index) : name(name), index(index) {}

  std::string getName() const { return name; }
  int getIndex() const { return index; }

  std::string toString() const override {
    return name + "[" + std::to_string(index) + "]";
  }
  void prettyPrint(std::ostream& f, int depth) const override;

  ExpressionValue getExprValue() const override { return false; }
};

class UnaryExpr : public Expression {
  UnaryOp op;
  std::unique_ptr<Expression> expr;

public:
  UnaryExpr(UnaryOp op, std::unique_ptr<Expression> expr)
      : op(op), expr(std::move(expr)) {}
  std::string toString() const override { return "UnaryExpr"; }
  void prettyPrint(std::ostream& f, int depth) const override;

  UnaryOp getOp() const { return op; }
  const Expression& getExpr() const { return *expr; }

  ExpressionValue getExprValue() const override {
    auto exprValue = expr->getExprValue();
    if (!exprValue.isConstant)
      return false;
    switch (op) {
    case UnaryOp::Positive:
      return exprValue.value;
    case UnaryOp::Negative:
      return -exprValue.value;
    default:
      return false;
    }
  }
};

class BinaryExpr : public Expression {
  BinaryOp op;
  std::unique_ptr<Expression> lhs, rhs;

public:
  BinaryExpr(BinaryOp op, std::unique_ptr<Expression> lhs,
             std::unique_ptr<Expression> rhs)
      : op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  std::string toString() const override { return "BinaryExpr"; }
  void prettyPrint(std::ostream& f, int depth) const override;

  const Expression& getLHS() const { return *lhs; }
  const Expression& getRHS() const { return *rhs; }
  BinaryOp getOp() const { return op; }
  ExpressionValue getExprValue() const override {
    auto lhsValue = lhs->getExprValue();
    auto rhsValue = rhs->getExprValue();
    if (!lhsValue.isConstant || !rhsValue.isConstant) {
      return false;
    }
    // both are constant
    switch (op) {
    case BinaryOp::Add:
      return lhsValue.value + rhsValue.value;
    case BinaryOp::Sub:
      return lhsValue.value - rhsValue.value;
    case BinaryOp::Mul:
      return lhsValue.value * rhsValue.value;
    case BinaryOp::Div:
      return lhsValue.value / rhsValue.value;
    default:
      return false;
    }
  }
};

class IfThenElseStmt : public Statement {
  std::unique_ptr<Expression> ifExpr;
  std::vector<std::unique_ptr<Statement>> thenBody;
  std::vector<std::unique_ptr<Statement>> elseBody;

public:
  IfThenElseStmt(std::unique_ptr<Expression> ifExpr)
      : ifExpr(std::move(ifExpr)) {}

  std::string toString() const override { return "IfThenElseStmt"; }

  void prettyPrint(std::ostream& f, int depth) const override;

  void addThenBody(std::unique_ptr<Statement> stmt) {
    thenBody.push_back(std::move(stmt));
  }

  void addElseBody(std::unique_ptr<Statement> stmt) {
    elseBody.push_back(std::move(stmt));
  }
};

class VersionStmt : public Statement {
  std::string version;

public:
  VersionStmt(std::string version) : version(version) {}
  std::string getVersion() const { return version; }

  std::string toString() const override { return "Version(" + version + ")"; }

  void prettyPrint(std::ostream& f, int depth) const override;
};

class IncludeStmt : public Statement {
  std::string fileName;

public:
  IncludeStmt(const std::string& fileName) : fileName(fileName) {}
  std::string getFileName() const { return fileName; }

  std::string toString() const override { return "Include(" + fileName + ")"; }

  void prettyPrint(std::ostream& f, int depth) const override {}
};

class QRegStmt : public Statement {
  std::string name;
  int size;

public:
  QRegStmt(std::string name, int size) : name(name), size(size) {}

  std::string getName() const { return name; }
  int getSize() const { return size; }

  std::string toString() const override {
    return "QReg(" + name + ", " + std::to_string(size) + ")";
  }

  void prettyPrint(std::ostream& f, int depth) const override;
};

class CRegStmt : public Statement {
  std::string name;
  int size;

public:
  CRegStmt(std::string name, int size) : name(name), size(size) {}

  std::string getName() const { return name; }
  int getSize() const { return size; }

  std::string toString() const override {
    return "CReg(" + name + ", " + std::to_string(size) + ")";
  }

  void prettyPrint(std::ostream& f, int depth) const override;
};

class GateApplyStmt : public Statement {
public:
  std::string name;
  std::vector<std::unique_ptr<Expression>> parameters;
  std::vector<std::unique_ptr<SubscriptExpr>> targets;

  GateApplyStmt(std::string name) : name(name) {}

  void addParameter(std::unique_ptr<Expression> param) {
    parameters.push_back(std::move(param));
  }

  void addTarget(std::unique_ptr<SubscriptExpr> targ) {
    targets.push_back(std::move(targ));
  }

  std::string toString() const override { return "gate " + name; }

  void prettyPrint(std::ostream& f, int depth) const override;
};

class RootNode : public Node {
public:
  std::vector<std::unique_ptr<Statement>> stmts;
  std::string toString() const override { return "Root"; }
  void prettyPrint(std::ostream& f, int depth) const override;
  void addStmt(std::unique_ptr<Statement> stmt) {
    stmts.push_back(std::move(stmt));
  }

  size_t countStmts() { return stmts.size(); }
  Statement getStmt(size_t index) { return *(stmts[index]); }

  void toLegacyCircuitGraph(cast::LegacyCircuitGraph&) const;
};

} // namespace openqasm::ast

#endif // OPENQASM_AST_H
