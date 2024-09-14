#ifndef QUENCH_AST_H
#define QUENCH_AST_H

#include <vector>
#include <memory>
#include <iostream>
#include <cassert>
#include <map>

#include "quench/QuantumGate.h"

namespace quench::circuit_graph {
    class CircuitGraph;
}

namespace quench::ast {

class Node {
public:
    virtual ~Node() = default;
    virtual std::ostream& print(std::ostream& os) const = 0;
};

class Expression : public Node {};

class Statement : public Node {};
class CircuitCompatibleStmt : public Statement {};

class GateApplyStmt : public CircuitCompatibleStmt {
public:
    std::string name;
    std::vector<int> qubits;
    std::vector<quantum_gate::GateParameter> params;
    int paramRefNumber;

    GateApplyStmt(const std::string& name, int paramRefNumber = -1)
        : name(name), qubits(), paramRefNumber(paramRefNumber) {}

    GateApplyStmt(const std::string& name, int paramRefNumber,
                  std::initializer_list<int> qubits)
        : name(name), qubits(qubits), paramRefNumber(paramRefNumber) {} 

    std::ostream& print(std::ostream& os) const override;
};

class GateChainStmt : public CircuitCompatibleStmt {
public:
    std::vector<GateApplyStmt> gates;

    GateChainStmt() : gates() {}

    std::ostream& print(std::ostream& os) const override;
};

class CircuitStmt : public Statement {
public:
    std::string name;
    int nqubits;
    int nparams;
    std::vector<std::unique_ptr<CircuitCompatibleStmt>> stmts;
    
    CircuitStmt() : nqubits(0), nparams(0), stmts() {}

    void addGateChain(const GateChainStmt& chain);

    std::ostream& print(std::ostream& os) const override;
};

/// @brief '#'<number:int> '=' '{' ... '}'';'
class ParameterDefStmt : public Statement {
public:
    int refNumber;
    quantum_gate::GateMatrix gateMatrix;

    ParameterDefStmt(int refNumber) : refNumber(refNumber), gateMatrix() {}

    std::ostream& print(std::ostream& os) const override;
};

class RootNode : public Node {
private:
    struct param_def_stmt_cmp {
        bool operator()
                (const ParameterDefStmt& a, const ParameterDefStmt& b) const {
            return a.refNumber < b.refNumber;
        }
    };
public:
    CircuitStmt circuit;
    std::vector<ParameterDefStmt> paramDefs;
    cas::Context casContext;

    RootNode() : circuit(), paramDefs(), casContext() {}

    std::ostream& print(std::ostream& os) const override;

    quench::circuit_graph::CircuitGraph toCircuitGraph() const;
};


} // namespace quench::ast

#endif // QUENCH_AST_H