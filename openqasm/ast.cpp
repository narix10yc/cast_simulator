#include "openqasm/ast.h"
#include "cast/LegacyCircuitGraph.h"

#include "llvm/ADT/SmallVector.h"

using namespace openqasm::ast;

void RootNode::toLegacyCircuitGraph(cast::LegacyCircuitGraph& graph) const {
  llvm::SmallVector<int> qubits;
  for (const auto& s : stmts) {
    cast::LegacyGateMatrix::gate_params_t params;
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
    auto matrix = cast::LegacyGateMatrix::FromName(gateApply->name, params);
    graph.appendGate(std::make_shared<cast::LegacyQuantumGate>(matrix, qubits));
  }
}