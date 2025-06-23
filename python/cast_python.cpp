#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"

#include "cast/CPU/CPUKernelManager.h"
#include "cast/CPU/CPUDensityMatrix.h"
#include "cast/Transform/Transform.h"
#include "openqasm/parser.h"

namespace py = pybind11;

namespace {

void bind_QuantumGate(py::module_& m) {
  // Base class: QuantumGate
  py::class_<cast::QuantumGate, cast::QuantumGatePtr>(m, "QuantumGate")
    .def_property_readonly("num_qubits", &cast::QuantumGate::nQubits)
    .def_property_readonly("qubits", 
      [](const cast::QuantumGate& self) {
        return self.qubits();
      }
    )
    .def("op_count", &cast::QuantumGate::opCount, py::arg("zero_tol"))
    .def("get_superop_gate", &cast::QuantumGate::getSuperopGate)
    .def("get_info", 
      [](const cast::QuantumGate& self, int verbose) {
        std::ostringstream oss;
        self.displayInfo(oss, verbose);
        return oss.str();
      },
      py::arg("verbose") = 1
    );

  // Derived class: StandardQuantumGate
  py::class_<cast::StandardQuantumGate,
              cast::QuantumGate,
              cast::StandardQuantumGatePtr>(m, "StandardQuantumGate")
    .def_static("random_unitary",
      [](const std::vector<int>& qubits) {
        return cast::StandardQuantumGate::RandomUnitary(qubits);
      },
      py::arg("qubits")
    )
    .def_static("I1", &cast::StandardQuantumGate::I1, py::arg("q"))
    .def_static("H", &cast::StandardQuantumGate::H, py::arg("q"))
    .def("set_noise_spc",
      &cast::StandardQuantumGate::setNoiseSPC,
      py::arg("p")
    )
    .def("get_superop_gate", &cast::StandardQuantumGate::getSuperopGate)
    .def("get_info",
      [](const cast::StandardQuantumGate& self, int verbose) {
        std::ostringstream oss;
        self.displayInfo(oss, verbose);
        return oss.str();
    },
    py::arg("verbose") = 1
  );

  // Derived class: SuperopQuantumGate
  py::class_<cast::SuperopQuantumGate,
             cast::QuantumGate,
             cast::SuperopQuantumGatePtr>(m, "SuperopQuantumGate")
    .def("get_info",
      [](const cast::SuperopQuantumGate& self, int verbose) {
        std::ostringstream oss;
        self.displayInfo(oss, verbose);
        return oss.str();
    },
    py::arg("verbose") = 1
  );
}

// Bind cast::ir::CircuitGraphNode to CircuitGraph in Python
void bind_CircuitGraph(py::module_& m) {
  using CircuitGraphNode = cast::ir::CircuitGraphNode;
  py::class_<CircuitGraphNode>(m, "CircuitGraph")
    .def("__str__", [](const CircuitGraphNode& self) {
      std::ostringstream oss;
      self.print(oss, 0);
      return oss.str();
    })
    .def("get_all_gates", &CircuitGraphNode::getAllGates)
    .def("get_visualization", [](const CircuitGraphNode& self, int verbose=1) {
      std::ostringstream oss;
      self.visualize(oss, verbose);
      return oss.str();
    },
    py::arg("verbose") = 1
    );
}

// Bind cast::ir::CircuitNode to Circuit in Python
void bind_Circuit(py::module_& m) {
  py::class_<cast::ir::CircuitNode>(m, "Circuit")
    .def("__str__", [](const cast::ir::CircuitNode& self) {
      std::ostringstream oss;
      self.print(oss, 0);
      return oss.str();
    })
    .def("get_all_circuit_graphs",
      &cast::ir::CircuitNode::getAllCircuitGraphs,
      py::return_value_policy::reference_internal
    );
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

void bind_CPUStatevector(py::module_& m) {
  py::class_<cast::CPUStatevectorF32>(m, "CPUStatevectorF32")
    .def(py::init<int, int>(), py::arg("num_qubits"), py::arg("simd_s"))
    .def("__getitem__", [](const cast::CPUStatevectorF32& self, size_t idx) {
      if (idx >= self.getN()) {
        throw std::out_of_range("Index out of range in CPUStatevectorF32");
      }
      return self.amp(idx);
    }, py::arg("idx")
    )
    .def_property_readonly("num_qubits",
      [](const cast::CPUStatevectorF32& self) {
        return self.nQubits();
      }
    )
    .def("normSquared", &cast::CPUStatevectorF32::normSquared)
    .def("norm", &cast::CPUStatevectorF32::norm)
    .def("probability",
      &cast::CPUStatevectorF32::prob, 
      py::arg("qubit")
    )
    .def("initialize",
      &cast::CPUStatevectorF32::initialize, py::arg("num_threads") = 1)
    .def("normalize",
      &cast::CPUStatevectorF32::normalize, py::arg("num_threads") = 1)
    .def("randomize",
      &cast::CPUStatevectorF32::randomize, py::arg("num_threads") = 1);
  
  py::class_<cast::CPUStatevectorF64>(m, "CPUStatevectorF64")
    .def(py::init<int, int>(), py::arg("num_qubits"), py::arg("simd_s"))
    .def("__getitem__", [](const cast::CPUStatevectorF64& self, size_t idx) {
      if (idx >= self.getN()) {
        throw std::out_of_range("Index out of range in CPUStatevectorF64");
      }
      return self.amp(idx);
    }, py::arg("idx")
    )
    .def_property_readonly("num_qubits",
      [](const cast::CPUStatevectorF32& self) {
        return self.nQubits();
      }
    )
    .def("normSquared", &cast::CPUStatevectorF64::normSquared)
    .def("norm", &cast::CPUStatevectorF64::norm)
    .def("probability",
      &cast::CPUStatevectorF64::prob, 
      py::arg("qubit")
    )
    .def("initialize",
      &cast::CPUStatevectorF64::initialize, py::arg("nThreads") = 1)
    .def("normalize",
      &cast::CPUStatevectorF64::normalize, py::arg("nThreads") = 1)
    .def("randomize",
      &cast::CPUStatevectorF64::randomize, py::arg("nThreads") = 1);
}

void bind_CPUKernelManager(py::module_& m) {
  py::enum_<cast::MatrixLoadMode>(m, "MatrixLoadMode")
    .value("UseMatImmValues", cast::MatrixLoadMode::UseMatImmValues)
    .value("StackLoadMatElems", cast::MatrixLoadMode::StackLoadMatElems)
    .export_values();

  py::class_<cast::CPUKernelGenConfig>(m, "CPUKernelGenConfig")
    .def(py::init<>())
    .def_readwrite("simd_s", &cast::CPUKernelGenConfig::simd_s)
    .def_readwrite("precision", &cast::CPUKernelGenConfig::precision)
    .def_readwrite("ampFormat", &cast::CPUKernelGenConfig::ampFormat)
    .def_readwrite("useFMA", &cast::CPUKernelGenConfig::useFMA)
    .def_readwrite("useFMS", &cast::CPUKernelGenConfig::useFMS)
    .def_readwrite("usePDEP", &cast::CPUKernelGenConfig::usePDEP)
    .def_readwrite("zeroTol", &cast::CPUKernelGenConfig::zeroTol)
    .def_readwrite("oneTol", &cast::CPUKernelGenConfig::oneTol)
    .def_readwrite("matrixLoadMode", &cast::CPUKernelGenConfig::matrixLoadMode)
    .def("get_info", [](const cast::CPUKernelGenConfig& self) {
      std::ostringstream oss;
      self.displayInfo(oss);
      return oss.str();
    });
  
  py::class_<cast::CPUKernelInfo>(m, "CPUKernelInfo")
    .def(py::init<>())
    .def_readonly("precision", &cast::CPUKernelInfo::precision)
    .def_readonly("llvm_func_name", &cast::CPUKernelInfo::llvmFuncName)
    .def_readonly("gate", &cast::CPUKernelInfo::gate)
    .def_property_readonly("executable",
      [](const cast::CPUKernelInfo& self) {
        return self.executable ? true : false;
      }
    )
    .def("get_info", [](const cast::CPUKernelInfo& self) {
        std::ostringstream oss;
        oss << "=== Info of CPUKernel @ " << (void*)&self << " ===\n";
        oss << "- Precision: " << self.precision << "\n";
        oss << "- LLVM Function Name: " << self.llvmFuncName << "\n";
        oss << "- Gate: " << (void*)(self.gate.get()) << "\n";
        oss << "- Executable: " << (self.executable ? "Yes" : "No") << "\n";
        return oss.str();
      }
    );

  py::class_<cast::CPUKernelManager>(m, "CPUKernelManager")
    .def(py::init<>())
    .def("get_info", [](const cast::CPUKernelManager& self) {
        std::ostringstream oss;
        self.displayInfo(oss);
        return oss.str();
      })
    .def("gen_cpu_gate", [](cast::CPUKernelManager& self, 
                            const cast::CPUKernelGenConfig& config, 
                            const cast::QuantumGatePtr& gate, 
                            const std::string& func_name) {
        auto result = self.genStandaloneGate(config, gate, func_name);
        if (!result) {
          throw std::runtime_error(
            "Failed to generate CPU gate. Reason: " + result.takeError());
        }
      },
      py::arg("config"), py::arg("gate"), py::arg("func_name")
    )
    .def("gen_cpu_gates_from_graph", [](cast::CPUKernelManager& self, 
                                        const cast::CPUKernelGenConfig& config, 
                                        const cast::ir::CircuitGraphNode& graph, 
                                        const std::string& graph_name) {
        auto rst = self.genCPUGatesFromGraph(config, graph, graph_name);
        if (!rst) {
          throw std::runtime_error(
            "Failed to generate CPU gates from graph. Reason: " +
            rst.takeError()
          );
        }
      },
      py::arg("config"), py::arg("graph"), py::arg("graph_name")
    )
    .def("get_kernel_by_name",
      [](const cast::CPUKernelManager& self, 
         const std::string& funcName){
        return self.getKernelByName(funcName);
      },
      py::arg("func_name")
    )
    .def("init_jit", [](cast::CPUKernelManager& self,
                        int nThreads, int optLevel, 
                        bool useLazyJit, int verbose) {
        llvm::OptimizationLevel llvmOptLevel = llvm::OptimizationLevel::O0;
        switch (optLevel) {
          case 0: llvmOptLevel = llvm::OptimizationLevel::O0; break;
          case 1: llvmOptLevel = llvm::OptimizationLevel::O1; break;
          case 2: llvmOptLevel = llvm::OptimizationLevel::O2; break;
          case 3: llvmOptLevel = llvm::OptimizationLevel::O3; break;
          // we default it to O3 anyways
          default: llvmOptLevel = llvm::OptimizationLevel::O3; break;
        }
        auto rst = self.initJIT(nThreads, llvmOptLevel, useLazyJit, verbose);
        if (!rst) {
          throw std::runtime_error(
            "Failed to initialize JIT: " + rst.takeError());
        }
      },
      py::arg("num_threads") = 1,
      py::arg("opt_level") = 1,
      py::arg("use_lazy_jit") = false,
      py::arg("verbose") = 0
    )
    .def("apply_gate_f32", 
      [](cast::CPUKernelManager& self, 
         cast::CPUStatevectorF32& sv, 
         const cast::CPUKernelInfo& kernel,
         int nThreads) {
        if (kernel.precision != 32) {
          throw std::runtime_error(
            "Kernel precision mismatch: expected 32, got " + 
            std::to_string(kernel.precision));
        }
        auto rst = self.applyCPUKernel(
          sv.data(), sv.nQubits(), kernel, nThreads); 
        if (!rst) {
          throw std::runtime_error(
            "Failed to apply gate. Reason" + rst.takeError());
        }
      },
      py::arg("sv"), py::arg("gate"), py::arg("num_threads") = 1
    )
    .def("apply_gate_f64", 
      [](cast::CPUKernelManager& self, 
         cast::CPUStatevectorF64& sv, 
         const cast::CPUKernelInfo& kernel,
         int nThreads) {
        if (kernel.precision != 64) {
          throw std::runtime_error(
            "Kernel precision mismatch: expected 32, got " + 
            std::to_string(kernel.precision));
        }
        auto rst = self.applyCPUKernel(
          sv.data(), sv.nQubits(), kernel, nThreads); 
        if (!rst) {
          throw std::runtime_error("Failed to apply gate: " + rst.takeError());
        }
      },
      py::arg("sv"), py::arg("gate"), py::arg("num_threads") = 1
    );
}

} // end of anonymous namespace

PYBIND11_MODULE(cast_python, m) {
  m.doc() = "Python bindings for the cast library";
  bind_QuantumGate(m);
  bind_CircuitGraph(m);
  bind_Circuit(m);
  bind_parse_circuit_from_qasm_file(m);

  bind_CPUStatevector(m);
  bind_CPUKernelManager(m);
}