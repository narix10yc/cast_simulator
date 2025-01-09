#include "openqasm/ast.h"
#include "saot/CircuitGraph.h"

#include "llvm/ADT/SmallVector.h"

using namespace openqasm::ast;

void RootNode::toCircuitGraph(saot::CircuitGraph& graph) const {
  llvm::SmallVector<int> qubits;
  for (const auto& s : stmts) {
    saot::GateMatrix::gate_params_t params;
    int i = 0;
    auto gateApply = dynamic_cast<GateApplyStmt*>(s.get());
    if (gateApply == nullptr) {
      // std::cerr << "skipping " << s->toString() << "\n";
      continue;
    }
    qubits.clear();
    for (const auto& t : gateApply->targets) {
      qubits.push_back(static_cast<unsigned>(t->getIndex()));
    }
    for (const auto& p : gateApply->parameters) {
      auto ev = p->getExprValue();
      assert(ev.isConstant);
      params[i++] = ev.value;
    }
    if (gateApply->name == "u3") {
      // Our representation of theta is 0.5 times that in OpenQASM
      params[0].get<double>() *= 0.5;
    }
    auto matrix = saot::GateMatrix::FromName(gateApply->name, params);
    auto* quantumGate = graph.acquireQuantumGateForward(matrix, qubits);
    graph.appendGate(quantumGate);
  }
}