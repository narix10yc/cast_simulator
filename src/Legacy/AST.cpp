#include "cast/Legacy/AST.h"
#include "cast/Legacy/CircuitGraph.h"

#include "utils/iocolor.h"
#include "utils/utils.h"

using namespace IOColor;
using namespace cast::legacy;
using namespace cast::legacy::ast;

namespace {

std::ostream& printGateParameters(std::ostream& os,
                                  const GateMatrix::gate_params_t& params) {
  const auto visitor = [&os](const utils::PODVariant<int, double>& arg) {
    if (arg.is<int>())
      os << arg.get<int>();
    else if (arg.is<double>())
      os << std::setprecision(16) << arg.get<double>();
    else
      assert(!arg.holdingValue() && "Expect Monostate");
  };

  if (!params[0].holdingValue())
    return os;
  os << "(";
  visitor(params[0]);

  if (!params[1].holdingValue())
    return os << ")";
  os << ", ";
  visitor(params[1]);

  if (!params[2].holdingValue())
    return os << ")";
  os << ", ";
  visitor(params[2]);
  return os << ")";
}
} // anonymous namespace

std::ostream& MeasureStmt::print(std::ostream& os) const {
  os << "m ";
  utils::printArrayNoBrackets(os, llvm::ArrayRef(qubits), ' ');
  return os << "\n";
}

std::ostream& QuantumCircuit::print(std::ostream& os) const {
  os << "circuit<nQubits=" << nQubits << ", nParams=" << nParams << "> " << name
     << " {\n";
  for (const auto& s : chains)
    s->print(os);

  if (!paramDefs.empty()) {
    os << "\n";
    for (const auto& def : paramDefs)
      def.print(os);
  }

  return os << "}\n";
}

std::ostream& GateApplyStmt::print(std::ostream& os) const {
  os << name;

  // parameter
  if (argument.is<int>())
    os << "(#" << argument.get<int>() << ")";
  else if (argument.is<GateMatrix::gate_params_t>()) {
    auto& gateParams = argument.get<GateMatrix::gate_params_t>();
    printGateParameters(os, gateParams);
  }

  // target qubits
  os << " ";
  utils::printArrayNoBrackets(os, llvm::ArrayRef(qubits), ' ');
  return os;
}

std::ostream& GateChainStmt::print(std::ostream& os) const {
  const auto size = gates.size();
  if (size == 0)
    return os;
  os << "  ";
  for (size_t i = 0; i < size - 1; i++)
    gates[i].print(os) << "\n@ ";
  gates[size - 1].print(os) << ";\n";
  return os;
}

std::ostream& ParameterDefStmt::print(std::ostream& os) const {
  assert(false && "Not Implemented");
  return os;
  // os << "#" << refNumber << " = { ";
  // if (gateMatrix.isConstantMatrix()) {
  //     auto it = gateMatrix.cData().cbegin();
  //     utils::print_complex(os, *it);
  //     while (++it != gateMatrix.cData().cend())
  //         utils::print_complex(os << ", ", *it);
  //     return os << " }\n";
  // }

  // assert(gateMatrix.isParametrizedMatrix());
  // auto it = gateMatrix.pData().cbegin();
  // it->print(os);
  // while (++it != gateMatrix.pData().cend())
  //     it->print(os << ", ");
  // return os << " }\n";
}

void QuantumCircuit::addChainStmt(std::unique_ptr<GateChainStmt> chain) {
  for (const auto& gate : chain->gates) {
    for (const auto& q : gate.qubits) {
      if (this->nQubits <= q)
        this->nQubits = q + 1;
    }
  }
  chains.emplace_back(std::move(chain));
}

std::shared_ptr<QuantumGate> QuantumCircuit::gateApplyToLegacyQuantumGate(
    const GateApplyStmt& gaStmt) const {
  if (gaStmt.argument.is<int>()) {
    // gaStmt relies on parameter reference
    const auto refNumber = gaStmt.argument.get<int>();
    for (const auto& defStmt : paramDefs) {
      if (defStmt.refNumber == refNumber)
        return std::make_shared<QuantumGate>(*defStmt.gateMatrix,
                                             gaStmt.qubits);
    }
    assert(false && "Cannot find parameter def stmt");
    return nullptr;
  }

  if (gaStmt.argument.is<GateMatrix::gate_params_t>())
    return std::make_shared<QuantumGate>(
        GateMatrix::FromName(gaStmt.name,
                             gaStmt.argument.get<GateMatrix::gate_params_t>()),
        gaStmt.qubits);
  return std::make_shared<QuantumGate>(GateMatrix::FromName(gaStmt.name),
                                       gaStmt.qubits);
}

void QuantumCircuit::toLegacyCircuitGraph(legacy::CircuitGraph& graph) const {
  for (const auto& s : chains) {
    if (s->isNot(Statement::SK_GateChain)) {
      std::cerr << BOLDYELLOW("Warning: ")
                << "Unable to convert to GateChainStmt when calling "
                   "RootNode::tolegacy::CircuitGraph\n";
      continue;
    }
    const auto* chain = static_cast<const GateChainStmt*>(s.get());
    if (chain->gates.empty())
      continue;

    auto quGate = gateApplyToLegacyQuantumGate(chain->gates[0]);
    for (int i = 1; i < chain->gates.size(); i++) {
      quGate = std::make_shared<QuantumGate>(
          quGate->lmatmul(*gateApplyToLegacyQuantumGate(chain->gates[i])));
    }
    graph.appendGate(quGate);
  }
}

QuantumCircuit
QuantumCircuit::FromLegacyCircuitGraph(const legacy::CircuitGraph& graph) {
  const auto allBlocks = graph.getAllBlocks();
  QuantumCircuit qc;

  for (const auto* block : allBlocks) {
    const auto gateNodes = block->getOrderedGates();
    auto chainStmt = std::make_unique<GateChainStmt>();
    for (const auto* gateNode : gateNodes) {
      auto name = gateNode->quantumGate->getName();
      chainStmt->gates.emplace_back(
          name,
          GateApplyStmt::arg_t(gateNode->quantumGate->gateMatrix.gateParams),
          gateNode->quantumGate->qubits);
    }
    qc.addChainStmt(std::move(chainStmt));
  }

  return qc;
}