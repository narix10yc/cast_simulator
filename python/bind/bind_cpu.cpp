#include "cast/CPU/CPU.h"

#include <pybind11/detail/common.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>

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
  py::class_<cast::CPUStatevectorFP32>(m, "CPUStatevectorFP32")
      .def(py::init<int, cast::CPUSimdWidth>(),
           py::arg("num_qubits"),
           py::arg("simd_width"))
      .def(
          "__getitem__",
          [](const cast::CPUStatevectorFP32& self, size_t idx) {
            if (idx >= self.getN()) {
              throw std::out_of_range(
                  "Index out of range in CPUStatevectorFP32");
            }
            return self.amp(idx);
          },
          py::arg("idx"))
      .def_property_readonly(
          "num_qubits",
          [](const cast::CPUStatevectorFP32& self) { return self.nQubits(); })
      .def("normSquared", &cast::CPUStatevectorFP32::normSquared)
      .def("norm", &cast::CPUStatevectorFP32::norm)
      .def("probability", &cast::CPUStatevectorFP32::prob, py::arg("qubit"))
      .def("initialize",
           &cast::CPUStatevectorFP32::initialize,
           py::arg("num_threads") = 1)
      .def("normalize",
           &cast::CPUStatevectorFP32::normalize,
           py::arg("num_threads") = 1)
      .def("randomize",
           &cast::CPUStatevectorFP32::randomize,
           py::arg("num_threads") = 1);

  py::class_<cast::CPUStatevectorFP64>(m, "CPUStatevectorFP64")
      .def(py::init<int, cast::CPUSimdWidth>(),
           py::arg("num_qubits"),
           py::arg("simd_width"))
      .def(
          "__getitem__",
          [](const cast::CPUStatevectorFP64& self, size_t idx) {
            if (idx >= self.getN()) {
              throw std::out_of_range(
                  "Index out of range in CPUStatevectorFP64");
            }
            return self.amp(idx);
          },
          py::arg("idx"))
      .def_property_readonly(
          "num_qubits",
          [](const cast::CPUStatevectorFP32& self) { return self.nQubits(); })
      .def("normSquared", &cast::CPUStatevectorFP64::normSquared)
      .def("norm", &cast::CPUStatevectorFP64::norm)
      .def("probability", &cast::CPUStatevectorFP64::prob, py::arg("qubit"))
      .def("initialize",
           &cast::CPUStatevectorFP64::initialize,
           py::arg("num_threads") = 1)
      .def("normalize",
           &cast::CPUStatevectorFP64::normalize,
           py::arg("num_threads") = 1)
      .def("randomize",
           &cast::CPUStatevectorFP64::randomize,
           py::arg("num_threads") = 1);
}

void bind_CPUKernelManager(py::module_& m) {
  py::enum_<cast::CPUMatrixLoadMode>(m, "CPUMatrixLoadMode")
      .value("UseMatImmValues", cast::CPUMatrixLoadMode::UseMatImmValues)
      .value("StackLoadMatElems", cast::CPUMatrixLoadMode::StackLoadMatElems)
      .export_values();

  py::class_<cast::CPUKernelGenConfig>(m, "CPUKernelGenConfig")
      .def(py::init<cast::Precision>(), py::arg("precision"))
      .def_readwrite("simd_width", &cast::CPUKernelGenConfig::simdWidth)
      .def_readwrite("precision", &cast::CPUKernelGenConfig::precision)
      .def_readwrite("use_fma", &cast::CPUKernelGenConfig::useFMA)
      .def_readwrite("use_fms", &cast::CPUKernelGenConfig::useFMS)
      .def_readwrite("use_pdep", &cast::CPUKernelGenConfig::usePDEP)
      .def_readwrite("zero_tol", &cast::CPUKernelGenConfig::zeroTol)
      .def_readwrite("one_tol", &cast::CPUKernelGenConfig::oneTol)
      .def_readwrite("matrix_load_mode",
                     &cast::CPUKernelGenConfig::matrixLoadMode)
      .def(
          "get_info",
          [](const cast::CPUKernelGenConfig& self, int verbose) {
            std::ostringstream oss;
            self.displayInfo({oss, verbose});
            return oss.str();
          },
          py::arg("verbose") = 1);

  py::class_<cast::CPUKernelInfo>(m, "CPUKernelInfo")
      .def(py::init<>())
      .def_readonly("precision", &cast::CPUKernelInfo::precision)
      .def_readonly("llvm_func_name", &cast::CPUKernelInfo::llvmFuncName)
      .def_readonly("gate", &cast::CPUKernelInfo::gate)
      .def_property_readonly("has_executable",
                             [](const cast::CPUKernelInfo& self) {
                               return self.executable ? true : false;
                             })
      .def("get_jit_time", &cast::CPUKernelInfo::getJitTime)
      .def("get_exec_time", &cast::CPUKernelInfo::getExecTime)
      .def(
          "get_info",
          [](const cast::CPUKernelInfo& self, int verbose) {
            std::ostringstream oss;
            self.displayInfo({oss, verbose});
            return oss.str();
          },
          py::arg("verbose") = 1);

  py::class_<cast::CPUKernelManager>(m, "CPUKernelManager")
      .def(py::init<>())
      .def(
          "get_info",
          [](const cast::CPUKernelManager& self, int verbose) {
            std::ostringstream oss;
            self.displayInfo({oss, verbose});
            return oss.str();
          },
          py::arg("verbose") = 1)
      .def(
          "gen_gate",
          [](cast::CPUKernelManager& self,
             const cast::CPUKernelGenConfig& config,
             const cast::QuantumGatePtr& gate,
             const std::string& func_name) {
            if (auto e = self.genGate(config, gate, func_name)) {
              throw std::runtime_error("Failed to generate gate: " +
                                       llvm::toString(std::move(e)));
            }
          },
          py::arg("config"),
          py::arg("gate"),
          py::arg("func_name") = "")
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
          [](cast::CPUKernelManager& self, const std::string& funcName) {
            return self.getKernelByName(funcName);
          },
          py::arg("func_name"))
      .def("compile_default_pool",
           [](cast::CPUKernelManager& self, int optLevel) {
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
             if (auto e = self.compileDefaultPool(llvmOptLevel)) {
               throw std::runtime_error("Failed to compile default pool: " +
                                        llvm::toString(std::move(e)));
             }
           })
      .def("compile_pool",
           [](cast::CPUKernelManager& self,
              const std::string& poolName,
              int optLevel) {
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
             if (auto e = self.compilePool(poolName, llvmOptLevel)) {
               throw std::runtime_error("Failed to compile pool '" + poolName +
                                        "': " + llvm::toString(std::move(e)));
             }
           })
      .def(
          "compile_all",
          [](cast::CPUKernelManager& self,
             int nThreads,
             int optLevel,
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
            if (auto e = self.compileAll(llvmOptLevel, verbose)) {
              throw std::runtime_error("Failed to compile all pools: " +
                                       llvm::toString(std::move(e)));
            }
          },
          py::arg("num_threads") = 1,
          py::arg("opt_level") = 1,
          py::arg("verbose") = 0)
      .def(
          "get_kernels_in_pool",
          [](const cast::CPUKernelManager& self,
             const std::string& poolName) -> py::list {
            if (self.pools().find(poolName) == self.pools().end()) {
              throw std::runtime_error("Kernel pool not found: " + poolName);
            }
            const auto& pool = self.pools().at(poolName);

            py::list out;

            py::handle parent = py::cast(&self);

            for (const auto& item : pool.items()) {
              out.append(py::cast(
                  *item.kernel, py::return_value_policy::reference, parent));
            }
            return out;
          },
          py::arg("pool_name"),
          py::return_value_policy::reference_internal)
      .def(
          "apply_kernel_f32",
          [](cast::CPUKernelManager& self,
             cast::CPUStatevectorFP32& sv,
             cast::CPUKernelInfo& kernel,
             int nThreads) {
            if (kernel.precision != cast::Precision::FP32) {
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
             cast::CPUStatevectorFP64& sv,
             cast::CPUKernelInfo& kernel,
             int nThreads) {
            if (kernel.precision != cast::Precision::FP64) {
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

void bind_CPUOptimizer(py::module_& m) {
  py::class_<cast::CPUOptimizer, cast::OptimizerBase>(m, "CPUOptimizer")
      .def(py::init<>())
      .def("enable_fusion",
           &cast::CPUOptimizer::enableFusion,
           py::arg("enable") = true)
      .def("enable_canonicalization",
           &cast::CPUOptimizer::enableCanonicalization,
           py::arg("enable") = true)
      .def("enable_cfo",
           &cast::CPUOptimizer::enableCFO,
           py::arg("enable") = true)
      .def("set_sizeonly_fusion_config",
           &cast::CPUOptimizer::setSizeOnlyFusionConfig,
           py::arg("size"))
      .def(
          "get_info",
          [](const cast::CPUOptimizer& self, int verbose) {
            std::ostringstream oss;
            self.displayInfo({oss, verbose});
            return oss.str();
          },
          py::arg("verbose") = 1)
      .def(
          "run",
          [](const cast::CPUOptimizer& self,
             cast::ir::CircuitGraphNode& graph,
             int verbose) {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect;

            utils::Logger logger(std::cout, verbose);
            self.run(graph, logger);
          },
          py::arg("graph"),
          py::arg("verbose") = 1);
}

} // end of anonymous namespace

void bind_cpu(py::module_& m) {
  bind_simdWidth(m);
  bind_CPUStatevector(m);
  bind_CPUKernelManager(m);
  bind_CPUOptimizer(m);
}