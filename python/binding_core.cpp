#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "cast/Core/Optimizer.h"
#include "cast/Transform/Transform.h"
#include "openqasm/Parser.h"

namespace py = pybind11;

namespace {

void bind_QuantumGate(py::module_& m) {
  // Base class: QuantumGate
  py::class_<cast::QuantumGate, cast::QuantumGatePtr>(m, "QuantumGate")
      .def_property_readonly("num_qubits", &cast::QuantumGate::nQubits)
      .def_property_readonly(
          "qubits", [](const cast::QuantumGate& self) { return self.qubits(); })
      .def("op_count", &cast::QuantumGate::opCount, py::arg("zero_tol"))
      .def("get_superop_gate", &cast::QuantumGate::getSuperopGate)
      .def(
          "get_info",
          [](const cast::QuantumGate& self, int verbose) {
            std::ostringstream oss;
            self.displayInfo(oss, verbose);
            return oss.str();
          },
          py::arg("verbose") = 1);

  // Derived class: StandardQuantumGate
  py::class_<cast::StandardQuantumGate,
             cast::QuantumGate,
             cast::StandardQuantumGatePtr>(m, "StandardQuantumGate")
      .def_static(
          "random_unitary",
          [](const std::vector<int>& qubits) {
            return cast::StandardQuantumGate::RandomUnitary(qubits);
          },
          py::arg("qubits"))
      .def_static("I1", &cast::StandardQuantumGate::I1, py::arg("q"))
      .def_static("H", &cast::StandardQuantumGate::H, py::arg("q"))
      .def("set_noise_spc",
           &cast::StandardQuantumGate::setNoiseSPC,
           py::arg("p"))
      .def("get_superop_gate", &cast::StandardQuantumGate::getSuperopGate)
      .def(
          "get_info",
          [](const cast::StandardQuantumGate& self, int verbose) {
            std::ostringstream oss;
            self.displayInfo(oss, verbose);
            return oss.str();
          },
          py::arg("verbose") = 1);

  // Derived class: SuperopQuantumGate
  py::class_<cast::SuperopQuantumGate,
             cast::QuantumGate,
             cast::SuperopQuantumGatePtr>(m, "SuperopQuantumGate")
      .def(
          "get_info",
          [](const cast::SuperopQuantumGate& self, int verbose) {
            std::ostringstream oss;
            self.displayInfo(oss, verbose);
            return oss.str();
          },
          py::arg("verbose") = 1);
}

// Bind cast::ir::CircuitGraphNode to CircuitGraph in Python
void bind_CircuitGraph(py::module_& m) {
  using CircuitGraphNode = cast::ir::CircuitGraphNode;
  py::class_<CircuitGraphNode>(m, "CircuitGraph")
      .def("__str__",
           [](const CircuitGraphNode& self) {
             std::ostringstream oss;
             self.print(oss, 0);
             return oss.str();
           })
      .def("get_all_gates", &CircuitGraphNode::getAllGates)
      .def("get_visualization", [](const CircuitGraphNode& self) {
        std::ostringstream oss;
        self.visualize(oss);
        return oss.str();
      });
}

void bind_Circuit(py::module_& m) {
  py::class_<cast::ir::CircuitNode>(m, "Circuit")
      .def("__str__",
           [](const cast::ir::CircuitNode& self) {
             std::ostringstream oss;
             self.print(oss, 0);
             return oss.str();
           })
      .def("get_all_circuit_graphs",
           &cast::ir::CircuitNode::getAllCircuitGraphs,
           py::return_value_policy::reference_internal);
}

void bind_parse_circuit_from_qasm_file(py::module_& m) {
  using namespace cast::transform;
  m.def("parse_circuit_from_qasm_file", [](const std::string& fileName) {
    openqasm::Parser qasmParser(fileName);
    auto qasmRoot = qasmParser.parse();
    cast::ast::ASTContext astCtx;
    auto astCircuit = cvtQasmCircuitToAstCircuit(*qasmRoot, astCtx);
    return cvtAstCircuitToIrCircuit(*astCircuit, astCtx);
  });
}

void bind_OptimizerBase(py::module_& m) {
  py::class_<cast::OptimizerBase>(m, "OptimizerBase")
      .def(
          "run_circuit",
          [](const cast::OptimizerBase& self,
             cast::ir::CircuitNode& circuit,
             int verbose) {
            utils::Logger logger(std::cerr, verbose);
            self.run(circuit, logger);
          },
          py::arg("circuit"),
          py::arg("verbose") = 1)
      .def(
          "run_circuit_graph",
          [](const cast::OptimizerBase& self,
             cast::ir::CircuitGraphNode& graph,
             int verbose) {
            utils::Logger logger(std::cerr, verbose);
            self.run(graph, logger);
          },
          py::arg("graph"),
          py::arg("verbose") = 1);
}

} // end of anonymous namespace

void bind_core(py::module_& m) {
  bind_QuantumGate(m);
  bind_CircuitGraph(m);
  bind_Circuit(m);
  bind_parse_circuit_from_qasm_file(m);
  bind_OptimizerBase(m);
}