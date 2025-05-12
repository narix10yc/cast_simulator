#ifndef CAST_AST_H
#define CAST_AST_H

#include "cast/QuantumGate.h"

namespace cast {
class CircuitGraph;
}

namespace cast::ast {

class Statement {
public:
  enum StatementKind {
    SK_Measure,
    SK_ParamDef,
    SK_GateApply,
    SK_GateChain,
  };
private:
  StatementKind _kind;
public:
  Statement(StatementKind kind) : _kind(kind) {}

  bool is(StatementKind kind) const { return _kind == kind; }
  bool isNot(StatementKind kind) const { return _kind != kind; }

  virtual ~Statement() = default;

  virtual std::ostream& print(std::ostream& os) const = 0;
};

class MeasureStmt : public Statement {
public:
  llvm::SmallVector<int> qubits;

  explicit MeasureStmt(int q) : Statement(SK_Measure), qubits({q}) {}

  MeasureStmt(std::initializer_list<int> qubits)
    : Statement(SK_Measure), qubits(qubits) {}

  explicit MeasureStmt(const llvm::SmallVector<int>& qubits)
    : Statement(SK_Measure), qubits(qubits) {}

  std::ostream& print(std::ostream& os) const override;
};

/// @brief '#'<number:int> '=' '{' ... '}'';'
class ParameterDefStmt : public Statement {
public:
  int refNumber;
  std::unique_ptr<LegacyGateMatrix> gateMatrix;

  ParameterDefStmt(int refNumber, std::unique_ptr<LegacyGateMatrix> gateMatrix)
    : Statement(SK_ParamDef)
    , refNumber(refNumber)
    , gateMatrix(std::move(gateMatrix)) {}

  std::ostream& print(std::ostream& os) const override;
};

class GateApplyStmt : public Statement {
public:
  using arg_t = utils::PODVariant<int, LegacyGateMatrix::gate_params_t>;
  std::string name;
  arg_t argument;
  llvm::SmallVector<int> qubits;

  GateApplyStmt(const std::string& name, const llvm::SmallVector<int>& qubits)
    : Statement(SK_GateApply), name(name), argument(), qubits(qubits) {}

  GateApplyStmt(
      const std::string& name, const arg_t& paramRefOrMatrix,
      const llvm::SmallVector<int>& qubits)
    : Statement(SK_GateApply)
    , name(name)
    , argument(paramRefOrMatrix)
    , qubits(qubits) {}

  std::ostream& print(std::ostream& os) const override;
};

class GateChainStmt : public Statement {
public:
  llvm::SmallVector<GateApplyStmt> gates;

  GateChainStmt() : Statement(SK_GateChain), gates() {}

  std::ostream& print(std::ostream& os) const override;
};

/// \c QuantumCircuit is the text representation of a quantum circuit. It starts
/// with the `circuit` keyword.
class QuantumCircuit {
public:
  std::string name;
  int nQubits;
  int nParams;
  std::vector<std::unique_ptr<GateChainStmt>> chains;
  std::vector<ParameterDefStmt> paramDefs;

  QuantumCircuit(const std::string& name = "qc")
      : name(name), nQubits(0), nParams(0), chains(), paramDefs() {
    chains.reserve(16);
  }

  std::ostream& print(std::ostream& os) const;

  void addChainStmt(std::unique_ptr<GateChainStmt> chain);

  std::shared_ptr<QuantumGate>
  gateApplyToQuantumGate(const GateApplyStmt&) const;

  // CircuitGraph forbids copy and moves.
  void toCircuitGraph(CircuitGraph&) const;
  static QuantumCircuit FromCircuitGraph(const CircuitGraph&);
};

} // namespace cast::ast

#endif // CAST_AST_H