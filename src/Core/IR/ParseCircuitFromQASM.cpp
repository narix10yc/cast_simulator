#include "cast/ADT/GateMatrix.h"
#include "cast/Core/IRNode.h"
#include "cast/Transform/Transform.h"
#include "openqasm/Parser.h"
#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>

using namespace cast;

llvm::Expected<std::unique_ptr<ir::CircuitNode>>
cast::parseCircuitFromQASMFile(const std::string& qasmFileName) {
  openqasm::Parser parser(qasmFileName);
  auto qasmRoot = parser.parse();
  if (qasmRoot == nullptr) {
    return llvm::createStringError("Failed to parse QASM file: " +
                                   qasmFileName);
  }
  ast::ASTContext astCtx;
  auto astCircuit = transform::cvtQasmCircuitToAstCircuit(*qasmRoot, astCtx);
  if (astCircuit == nullptr) {
    return llvm::createStringError("Failed to convert QASM AST to CAST AST: " +
                                   qasmFileName);
  }

  auto circuit = transform::cvtAstCircuitToIrCircuit(*astCircuit, astCtx);
  if (circuit == nullptr) {
    return llvm::createStringError("Failed to convert CAST AST to CAST IR: " +
                                   qasmFileName);
  }
  return circuit;
}

static ScalarGateMatrixPtr
scalarGMFromName(std::string_view nameView,
                 int nParameters,
                 const std::array<double, 3>& parameters,
                 const QuantumGate::TargetQubitsType& qubits) {
  // convert name to upper case
  std::string name(nameView);
  for (auto& c : name)
    c = std::toupper(c);

  if (name == "X") {
    assert(nParameters == 0);
    return ScalarGateMatrix::X();
  }
  if (name == "Y") {
    assert(nParameters == 0);
    return ScalarGateMatrix::Y();
  }
  if (name == "Z") {
    assert(nParameters == 0);
    return ScalarGateMatrix::Z();
  }
  if (name == "H") {
    assert(nParameters == 0);
    return ScalarGateMatrix::H();
  }
  if (name == "RX") {
    assert(nParameters == 1);
    return ScalarGateMatrix::RX(parameters[0]);
  }
  if (name == "RY") {
    assert(nParameters == 1);
    return ScalarGateMatrix::RY(parameters[0]);
  }
  if (name == "RZ") {
    assert(nParameters == 1);
    return ScalarGateMatrix::RZ(parameters[0]);
  }
  if (name == "U3") {
    assert(nParameters == 3);
    return ScalarGateMatrix::U1q(parameters[0], parameters[1], parameters[2]);
  }

  if (name == "CX" || name == "CNOT") {
    assert(nParameters == 0);
    return ScalarGateMatrix::CX();
  }
  if (name == "CP") {
    assert(nParameters == 1);
    return ScalarGateMatrix::CP(parameters[0]);
  }

  return nullptr;
}

llvm::Expected<std::unique_ptr<ir::CircuitGraphNode>>
cast::parseCircuitGraphFromQASMFile(const std::string& qasmFileName) {
  openqasm::Parser parser(qasmFileName);
  auto qasmRoot = parser.parse();
  if (qasmRoot == nullptr) {
    return llvm::createStringError("Failed to parse QASM file: " +
                                   qasmFileName);
  }

  auto cg = std::make_unique<ir::CircuitGraphNode>();
  std::array<double, 3> parameters;
  QuantumGate::TargetQubitsType qubits;
  for (const auto& s : qasmRoot->stmts) {
    auto* gateApplyStmt = llvm::dyn_cast<openqasm::ast::GateApplyStmt>(s.get());
    if (gateApplyStmt == nullptr)
      continue;

    // parameters
    int nParameters = gateApplyStmt->parameters.size();
    if (nParameters > 0) {
      auto ev = gateApplyStmt->parameters[0]->getExprValue();
      assert(ev.isConstant);
      parameters[0] = ev.value;
    }
    if (nParameters > 1) {
      auto ev = gateApplyStmt->parameters[1]->getExprValue();
      assert(ev.isConstant);
      parameters[1] = ev.value;
    }
    if (nParameters > 2) {
      auto ev = gateApplyStmt->parameters[2]->getExprValue();
      assert(ev.isConstant);
      parameters[2] = ev.value;
    }
    assert(nParameters <= 3);
    // target qubits
    qubits.clear();
    for (const auto& t : gateApplyStmt->targets)
      qubits.push_back(t->getIndex());
    auto scalarGM =
        scalarGMFromName(gateApplyStmt->name, nParameters, parameters, qubits);
    if (scalarGM == nullptr) {
      return llvm::createStringError("Unsupported gate '" +
                                     gateApplyStmt->name +
                                     "' in QASM file: " + qasmFileName);
    }
    auto qGate = StandardQuantumGate::Create(scalarGM, nullptr, qubits);
    cg->insertGate(std::move(qGate));
  }
  return cg;
}