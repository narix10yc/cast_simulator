#include "pybind11/pybind11.h"
#include "pybind11/complex.h"

#include "cast/CPU/CPUKernelManager.h"
#include "cast/CPU/CPUStatevector.h"

namespace py = pybind11;

namespace {

void bind_simdWidth(py::module_& m) {
  py::enum_<cast::CPUSimdWidth>(m, "CPUSimdWidth")
      .value("W0", cast::CPUSimdWidth::W0)
      .value("W64", cast::CPUSimdWidth::W64)
      .value("W128", cast::CPUSimdWidth::W128)
      .value("W256", cast::CPUSimdWidth::W256)
      .value("W512", cast::CPUSimdWidth::W512);
}

void bind_CPUStatevector(py::module_& m) {
  py::class_<cast::CPUStatevectorF32>(m, "CPUStatevectorF32")
      .def(py::init<int, cast::CPUSimdWidth>(),
           py::arg("num_qubits"),
           py::arg("simd_width"))
      .def(
          "__getitem__",
          [](const cast::CPUStatevectorF32& self, size_t idx) {
            if (idx >= self.getN()) {
              throw std::out_of_range(
                  "Index out of range in CPUStatevectorF32");
            }
            return self.amp(idx);
          },
          py::arg("idx"))
      .def_property_readonly(
          "num_qubits",
          [](const cast::CPUStatevectorF32& self) { return self.nQubits(); })
      .def("normSquared", &cast::CPUStatevectorF32::normSquared)
      .def("norm", &cast::CPUStatevectorF32::norm)
      .def("probability", &cast::CPUStatevectorF32::prob, py::arg("qubit"))
      .def("initialize",
           &cast::CPUStatevectorF32::initialize,
           py::arg("num_threads") = 1)
      .def("normalize",
           &cast::CPUStatevectorF32::normalize,
           py::arg("num_threads") = 1)
      .def("randomize",
           &cast::CPUStatevectorF32::randomize,
           py::arg("num_threads") = 1);

  py::class_<cast::CPUStatevectorF64>(m, "CPUStatevectorF64")
      .def(py::init<int, cast::CPUSimdWidth>(),
           py::arg("num_qubits"),
           py::arg("simd_width"))
      .def(
          "__getitem__",
          [](const cast::CPUStatevectorF64& self, size_t idx) {
            if (idx >= self.getN()) {
              throw std::out_of_range(
                  "Index out of range in CPUStatevectorF64");
            }
            return self.amp(idx);
          },
          py::arg("idx"))
      .def_property_readonly(
          "num_qubits",
          [](const cast::CPUStatevectorF32& self) { return self.nQubits(); })
      .def("normSquared", &cast::CPUStatevectorF64::normSquared)
      .def("norm", &cast::CPUStatevectorF64::norm)
      .def("probability", &cast::CPUStatevectorF64::prob, py::arg("qubit"))
      .def("initialize",
           &cast::CPUStatevectorF64::initialize,
           py::arg("nThreads") = 1)
      .def("normalize",
           &cast::CPUStatevectorF64::normalize,
           py::arg("nThreads") = 1)
      .def("randomize",
           &cast::CPUStatevectorF64::randomize,
           py::arg("nThreads") = 1);
}

void bind_CPUKernelManager(py::module_& m) {
  py::enum_<cast::CPUMatrixLoadMode>(m, "CPUMatrixLoadMode")
      .value("UseMatImmValues", cast::CPUMatrixLoadMode::UseMatImmValues)
      .value("StackLoadMatElems", cast::CPUMatrixLoadMode::StackLoadMatElems)
      .export_values();

  py::class_<cast::CPUKernelGenConfig>(m, "CPUKernelGenConfig")
      .def(py::init<>())
      .def_readwrite("simd_s", &cast::CPUKernelGenConfig::simdWidth)
      .def_readwrite("precision", &cast::CPUKernelGenConfig::precision)
      .def_readwrite("useFMA", &cast::CPUKernelGenConfig::useFMA)
      .def_readwrite("useFMS", &cast::CPUKernelGenConfig::useFMS)
      .def_readwrite("usePDEP", &cast::CPUKernelGenConfig::usePDEP)
      .def_readwrite("zeroTol", &cast::CPUKernelGenConfig::zeroTol)
      .def_readwrite("oneTol", &cast::CPUKernelGenConfig::oneTol)
      .def_readwrite("matrixLoadMode",
                     &cast::CPUKernelGenConfig::matrixLoadMode)
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
                             })
      .def("get_info", [](const cast::CPUKernelInfo& self) {
        std::ostringstream oss;
        self.displayInfo(oss);
        return oss.str();
      });

  py::class_<cast::CPUKernelManager>(m, "CPUKernelManager")
      .def(py::init<>())
      .def("get_info",
           [](const cast::CPUKernelManager& self) {
             std::ostringstream oss;
             self.displayInfo(oss);
             return oss.str();
           })
      .def(
          "gen_cpu_gate",
          [](cast::CPUKernelManager& self,
             const cast::CPUKernelGenConfig& config,
             const cast::QuantumGatePtr& gate,
             const std::string& func_name) {
            if (auto e = self.genStandaloneGate(config, gate, func_name)) {
              throw std::runtime_error("Failed to generate CPU gate: " +
                                       llvm::toString(std::move(e)));
            }
          },
          py::arg("config"),
          py::arg("gate"),
          py::arg("func_name"))
      .def(
          "gen_graph_gates",
          [](cast::CPUKernelManager& self,
             const cast::CPUKernelGenConfig& config,
             const cast::ir::CircuitGraphNode& graph,
             const std::string& graph_name) {
            if (auto e = self.genGraphGates(config, graph, graph_name)) {
              throw std::runtime_error(
                  "Failed to generate CPU gates from graph: " +
                  llvm::toString(std::move(e)));
            }
          },
          py::arg("config"),
          py::arg("graph"),
          py::arg("graph_name"))
      .def(
          "get_kernel_by_name",
          [](const cast::CPUKernelManager& self, const std::string& funcName) {
            return self.getKernelByName(funcName);
          },
          py::arg("func_name"))
      .def(
          "init_jit",
          [](cast::CPUKernelManager& self,
             int nThreads,
             int optLevel,
             bool useLazyJit,
             int verbose) {
            llvm::OptimizationLevel llvmOptLevel = llvm::OptimizationLevel::O0;
            switch (optLevel) {
            case 0:
              llvmOptLevel = llvm::OptimizationLevel::O0;
              break;
            case 1:
              llvmOptLevel = llvm::OptimizationLevel::O1;
              break;
            case 2:
              llvmOptLevel = llvm::OptimizationLevel::O2;
              break;
            case 3:
              llvmOptLevel = llvm::OptimizationLevel::O3;
              break;
            // we default it to O1
            default:
              llvmOptLevel = llvm::OptimizationLevel::O1;
              break;
            }
            if (auto e = self.initJIT(llvmOptLevel, useLazyJit, verbose)) {
              throw std::runtime_error("Failed to initialize JIT: " +
                                       llvm::toString(std::move(e)));
            }
          },
          py::arg("num_threads") = 1,
          py::arg("opt_level") = 1,
          py::arg("use_lazy_jit") = false,
          py::arg("verbose") = 0)
      .def(
          "apply_kernel_f32",
          [](cast::CPUKernelManager& self,
             cast::CPUStatevectorF32& sv,
             const cast::CPUKernelInfo& kernel,
             int nThreads) {
            if (kernel.precision != cast::Precision::F32) {
              throw std::runtime_error(
                  "Kernel precision mismatch: expected 32, got " +
                  std::to_string(static_cast<int>(kernel.precision)));
            }
            if (auto e = self.applyCPUKernel(
                    sv.data(), sv.nQubits(), kernel, nThreads)) {
              throw std::runtime_error("Failed to apply gate: " +
                                       llvm::toString(std::move(e)));
            }
          },
          py::arg("sv"),
          py::arg("gate"),
          py::arg("num_threads") = 1)
      .def(
          "apply_kernel_f64",
          [](cast::CPUKernelManager& self,
             cast::CPUStatevectorF64& sv,
             const cast::CPUKernelInfo& kernel,
             int nThreads) {
            if (kernel.precision != cast::Precision::F64) {
              throw std::runtime_error(
                  "Kernel precision mismatch: expected 64, got " +
                  std::to_string(static_cast<int>(kernel.precision)));
            }
            if (auto e = self.applyCPUKernel(
                    sv.data(), sv.nQubits(), kernel, nThreads)) {
              throw std::runtime_error("Failed to apply gate: " +
                                       llvm::toString(std::move(e)));
            }
          },
          py::arg("sv"),
          py::arg("gate"),
          py::arg("num_threads") = 1);
}

} // end of anonymous namespace

void bind_cpu(py::module_& m) {
  bind_simdWidth(m);
  bind_CPUStatevector(m);
  bind_CPUKernelManager(m);
}