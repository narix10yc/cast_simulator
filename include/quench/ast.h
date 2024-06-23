#ifndef QUENCH_AST_H
#define QUENCH_AST_H

#include <vector>
#include <memory>
#include <iostream>
#include <cassert>
#include <map>

#include "quench/GateMatrix.h"

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
    int paramRefNumber;

    GateApplyStmt(const std::string& name, int paramRefNumber = -1)
        : name(name), qubits(), paramRefNumber(paramRefNumber) {}

    GateApplyStmt(const std::string& name, int paramRefNumber,
                  std::initializer_list<int> qubits)
        : name(name), qubits(qubits), paramRefNumber(paramRefNumber) {} 

    std::ostream& print(std::ostream& os) const override;
};

class GateBlockStmt : public CircuitCompatibleStmt {
public:
    std::vector<GateApplyStmt> gates;

    GateBlockStmt() : gates() {}

    std::ostream& print(std::ostream& os) const override {
        auto it = gates.begin();
        while (it != gates.end()) {
            os << ((it == gates.begin()) ? "  " : "@ ");
            it->print(os) << "\n";
        }
        return os;
    }
};

class CircuitStmt : public Statement {
public:
    std::string name;
    int nqubits;
    std::vector<std::unique_ptr<CircuitCompatibleStmt>> stmts;
    std::vector<std::shared_ptr<cas::VariableNode>> parameters;
    
    CircuitStmt() : nqubits(0), stmts() {}

    void addGateChain(const GateBlockStmt& chain);

    std::ostream& print(std::ostream& os) const override;
};

/// @brief '#'<number:int> '=' '{' ... '}'';'
class ParameterDefStmt : public Statement {
public:
    int refNumber;
    cas::GateMatrix matrix;

    ParameterDefStmt(int refNumber)
        : refNumber(refNumber), matrix() {}

    std::ostream& print(std::ostream& os) const override {return os;}

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
    std::vector<std::unique_ptr<CircuitStmt>> circuits;
    std::map<int, cas::GateMatrix> matrices;

    RootNode() : circuits(), matrices() {}

    void addCircuit(std::unique_ptr<CircuitStmt> circuit) {
        circuits.push_back(std::move(circuit));
    }

    /// @brief Insert the ParameterDefStmt. Record refNumber.
    /// @return whether the insertion is successful. return false if the
    /// refNumber already exists (re-definition)
    bool addParameterDef(const ParameterDefStmt& def) {
        return matrices.insert(std::make_pair(def.refNumber, def.matrix)).second;
    }

    std::ostream& print(std::ostream& os) const override;
};


} // namespace quench::ast

#endif // QUENCH_AST_H