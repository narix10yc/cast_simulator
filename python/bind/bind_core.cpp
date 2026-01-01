#include "cast/Core/CostModel.h"
#include "cast/Core/IRNode.h"
#include "pybind11/pybind11.h"
#include <pybind11/iostream.h>

#include "cast/Core/Optimizer.h"
#include "cast/Transform/Transform.h"

namespace py = pybind11;

namespace {

void bind_precision(py::module_& m) {
  py::enum_<cast::Precision>(m, "Precision")
      .value("FP32", cast::Precision::FP32)
      .value("FP64", cast::Precision::FP64)
      .value("Unknown", cast::Precision::Unknown);
}

void bind_QuantumGate(py::module_& m) {
  // Base class: QuantumGate
  py::class_<cast::QuantumGate, cast::QuantumGatePtr>(m, "QuantumGate")
      .def_property_readonly("num_qubits", &cast::QuantumGate::nQubits)
      .def_property_readonly(
          "qubits", [](const cast::QuantumGate& self) { return self.qubits(); })
      .def("op_count", &cast::QuantumGate::opCount, py::arg("zero_tol"))
      // .def("get_superop_gate", &cast::QuantumGate::getSuperopGate)
      .def(
          "print_info",
          [](const cast::QuantumGate& self, int verbose) -> void {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect(std::cout);
            self.displayInfo({std::cout, verbose});
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
      // .def("get_superop_gate", &cast::StandardQuantumGate::getSuperopGate)
      .def(
          "print_info",
          [](const cast::StandardQuantumGate& self, int verbose) -> void {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect(std::cout);
            self.displayInfo({std::cout, verbose});
          },
          py::arg("verbose") = 1);

  // Derived class: SuperopQuantumGate
  py::class_<cast::SuperopQuantumGate,
             cast::QuantumGate,
             cast::SuperopQuantumGatePtr>(m, "SuperopQuantumGate")
      .def(
          "print_info",
          [](const cast::SuperopQuantumGate& self, int verbose) -> void {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect(std::cout);
            self.displayInfo({std::cout, verbose});
          },
          py::arg("verbose") = 1);
}

// Bind cast::ir::CircuitGraphNode to CircuitGraph in Python
void bind_CircuitGraph(py::module_& m) {
  using CircuitGraphNode = cast::ir::CircuitGraphNode;
  py::class_<CircuitGraphNode>(m, "CircuitGraph")
      .def("__str__",
           [](const CircuitGraphNode& self) -> std::string {
             std::ostringstream oss;
             self.print(oss, 0);
             return oss.str();
           })
      .def("num_qubits", &CircuitGraphNode::nQubits)
      .def("get_all_gates", &CircuitGraphNode::getAllGates)
      .def(
          "print_info",
          [](const CircuitGraphNode& self, int verbose) -> void {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect(std::cout);
            self.displayInfo({std::cout, verbose});
          },
          py::arg("verbose") = 1)
      .def("visualize", [](const CircuitGraphNode& self) -> void {
        self.visualize(std::cerr);
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

void bind_parse_circuitgraph_from_qasm_file(py::module_& m) {
  using namespace cast::transform;
  m.def("parse_circuitgraph_from_qasm_file", [](const std::string& fileName) {
    auto cgOrError = cast::parseCircuitGraphFromQASMFile(fileName);
    if (!cgOrError) {
      throw std::runtime_error(
          "Failed to parse circuit graph from QASM file: " +
          llvm::toString(cgOrError.takeError()));
    }
    return std::move(*cgOrError);
  });
}

void bind_CostModel(py::module_& m) {
  py::class_<cast::CostModel>(m, "CostModel");
}

void bind_OptimizerBase(py::module_& m) {
  py::class_<cast::OptimizerBase>(m, "OptimizerBase")
      .def(
          "run_circuit",
          [](const cast::OptimizerBase& self,
             cast::ir::CircuitNode& circuit,
             int verbose) {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect(std::cout);

            utils::Logger logger(std::cout, verbose);
            self.run(circuit, logger);
          },
          py::arg("circuit"),
          py::arg("verbose") = 1)
      .def(
          "run_circuitgraph",
          [](const cast::OptimizerBase& self,
             cast::ir::CircuitGraphNode& graph,
             int verbose) {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect(std::cout);

            utils::Logger logger(std::cout, verbose);
            self.run(graph, logger);
          },
          py::arg("graph"),
          py::arg("verbose") = 1);
}

} // end of anonymous namespace

void bind_core(py::module_& m) {
  bind_precision(m);
  bind_QuantumGate(m);
  bind_CircuitGraph(m);
  bind_Circuit(m);
  bind_parse_circuitgraph_from_qasm_file(m);
  bind_CostModel(m);
  bind_OptimizerBase(m);
}