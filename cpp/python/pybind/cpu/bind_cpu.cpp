#include "cast/CPU/CPU.h"

#include <llvm/Support/Error.h>

#include <pybind11/detail/common.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

llvm::OptimizationLevel mapIntToLLVMOptLevel(int optLevel) {
  switch (optLevel) {
  case 0:
    return llvm::OptimizationLevel::O0;
  case 1:
    return llvm::OptimizationLevel::O1;
  case 2:
    return llvm::OptimizationLevel::O2;
  case 3:
    return llvm::OptimizationLevel::O3;
  // we default it to O1
  default:
    return llvm::OptimizationLevel::O1;
  }
}

void bind_simdWidth(py::module_& m) {
  py::enum_<cast::CPUSimdWidth>(m, "CPUSimdWidth")
      .value("W0", cast::CPUSimdWidth::W0)
      .value("W64", cast::CPUSimdWidth::W64)
      .value("W128", cast::CPUSimdWidth::W128)
      .value("W256", cast::CPUSimdWidth::W256)
      .value("W512", cast::CPUSimdWidth::W512);
}

template <typename SVType>
void bind_CPUStatevector(py::module_& m, const char* pyName) {
  py::class_<SVType>(m, pyName)
      .def(py::init<int, cast::CPUSimdWidth>(),
           py::arg("num_qubits"),
           py::arg("simd_width"))
      .def(
          "__getitem__",
          [](const SVType& self, size_t idx) {
            if (idx >= self.getN())
              throw std::out_of_range("Index out of range");
            return self.amp(idx);
          },
          py::arg("idx"))
      .def_property_readonly(
          "num_qubits",
          [](const SVType& self) -> int { return self.nQubits(); })
      .def("normSquared", &SVType::normSquared)
      .def("norm", &SVType::norm)
      .def("probability", &SVType::prob, py::arg("qubit"))
      .def("initialize", &SVType::initialize, py::arg("num_threads") = 0)
      .def("normalize", &SVType::normalize, py::arg("num_threads") = 0)
      .def("randomize", &SVType::randomize, py::arg("num_threads") = 0);
}

void bind_CPUKernelManager(py::module_& m) {
  // CPUMatrixLoadMode
  py::enum_<cast::CPUMatrixLoadMode>(m, "CPUMatrixLoadMode")
      .value("UseMatImmValues", cast::CPUMatrixLoadMode::UseMatImmValues)
      .value("StackLoadMatElems", cast::CPUMatrixLoadMode::StackLoadMatElems);

  // CPUKernelGenConfig
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
          "print_info",
          [](const cast::CPUKernelGenConfig& self, int verbose) -> void {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect(std::cout);
            self.displayInfo({std::cout, verbose});
          },
          py::arg("verbose") = 1);

  // CPUKernelInfo
  py::class_<cast::CPUKernelInfo>(m, "CPUKernelInfo")
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
          "print_info",
          [](const cast::CPUKernelInfo& self, int verbose) -> void {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect(std::cout);
            self.displayInfo({std::cout, verbose});
          },
          py::arg("verbose") = 1);

  // CPUKernelManager
  py::class_<cast::CPUKernelManager>(m, "CPUKernelManager")
      .def(py::init<>())
      .def(
          "print_info",
          [](const cast::CPUKernelManager& self, int verbose) -> void {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect(std::cout);
            self.displayInfo({std::cout, verbose});
          },
          py::arg("verbose") = 1)
      .def(
          "gen_gate",
          [](cast::CPUKernelManager& self,
             const cast::CPUKernelGenConfig& config,
             const cast::QuantumGatePtr& gate,
             const std::string& func_name) -> const cast::CPUKernelInfo* {
            if (auto e = self.genGate(config, gate, func_name)) {
              throw std::runtime_error("Failed to generate gate: " +
                                       llvm::toString(std::move(e)));
            }
            const auto& items = self.getDefaultPool().items();
            if (items.empty()) {
              // should not happen -- just a safe guard
              return nullptr;
            }
            return items.back().kernel.get();
          },
          py::arg("config"),
          py::arg("gate"),
          py::arg("func_name") = "",
          py::return_value_policy::reference_internal)
      .def(
          "gen_graph_gates",
          [](cast::CPUKernelManager& self,
             const cast::CPUKernelGenConfig& config,
             const cast::ir::CircuitGraphNode& graph,
             const std::string& pool_name) {
            if (auto e = self.genGraphGates(config, graph, pool_name)) {
              throw std::runtime_error(
                  "Failed to generate CPU gates from graph: " +
                  llvm::toString(std::move(e)));
            }
          },
          py::arg("config"),
          py::arg("graph"),
          py::arg("pool_name"))
      .def(
          "get_ir",
          [](cast::CPUKernelManager& self,
             const std::string& funcName) -> std::string {
            auto ir = self.getIR(funcName);
            if (!ir)
              throw std::runtime_error(llvm::toString(ir.takeError()));
            return *ir;
          },
          py::arg("func_name"))
      .def(
          "get_kernel_by_name",
          [](cast::CPUKernelManager& self, const std::string& funcName) {
            return self.getKernelByName(funcName);
          },
          py::arg("func_name"))
      .def("compile_default_pool",
           [](cast::CPUKernelManager& self, int optLevel) {
             auto llvmOptLevel = mapIntToLLVMOptLevel(optLevel);
             if (auto e = self.compileDefaultPool(llvmOptLevel)) {
               throw std::runtime_error("Failed to compile default pool: " +
                                        llvm::toString(std::move(e)));
             }
           })
      .def("compile_pool",
           [](cast::CPUKernelManager& self,
              const std::string& poolName,
              int optLevel) {
             auto llvmOptLevel = mapIntToLLVMOptLevel(optLevel);
             if (auto e = self.compilePool(poolName, llvmOptLevel)) {
               throw std::runtime_error("Failed to compile pool '" + poolName +
                                        "': " + llvm::toString(std::move(e)));
             }
           })
      .def(
          "compile_all_pools",
          [](cast::CPUKernelManager& self, int optLevel, int verbose) {
            auto llvmOptLevel = mapIntToLLVMOptLevel(optLevel);
            if (auto e = self.compileAllPools(llvmOptLevel, verbose)) {
              throw std::runtime_error("Failed to compile all pools: " +
                                       llvm::toString(std::move(e)));
            }
          },
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
          "apply_kernel_fp32",
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
          "apply_kernel_fp64",
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

void bind_CPUPerformanceCache(py::module_& m) {
  py::class_<cast::CPUPerformanceCache>(m, "CPUPerformanceCache")
      .def(py::init<>())
      .def("load_from_file",
           &cast::CPUPerformanceCache::loadFromFile,
           py::arg("filename"))
      .def(
          "run_experiments",
          [](cast::CPUPerformanceCache& self,
             const cast::CPUKernelGenConfig& cpuConfig,
             int nQubits,
             int nThreads,
             int nRuns,
             int verbose) {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect_out(std::cout);
            py::scoped_ostream_redirect redirect_err(std::cerr);
            if (auto e = self.runExperiments(
                    cpuConfig, nQubits, nThreads, nRuns, verbose)) {
              throw std::runtime_error(
                  "Failed to run CPU performance experiments: " +
                  llvm::toString(std::move(e)));
            }
          },
          py::arg("cpu_config"),
          py::arg("n_qubits"),
          py::arg("n_threads"),
          py::arg("n_runs"),
          py::arg("verbose") = 1)
      .def("raw",
           [](const cast::CPUPerformanceCache& self) -> std::string {
             std::ostringstream oss;
             self.writeCache(oss);
             return oss.str();
           })
      .def(
          "save",
          [](const cast::CPUPerformanceCache& self,
             const std::string& filename,
             bool overwrite) {
            if (auto e = self.save(filename, overwrite)) {
              throw std::runtime_error(
                  "Failed to save CPU performance cache: " +
                  llvm::toString(std::move(e)));
            }
          },
          py::arg("filename"),
          py::arg("overwrite") = false);
}

void bind_CPUCostModel(py::module_& m) {
  py::class_<cast::CPUCostModel, cast::CostModel>(m, "CPUCostModel")
      .def(py::init<int, cast::Precision, double>(),
           py::arg("query_num_threads"),
           py::arg("query_precision"),
           py::arg("zero_tol") = 1e-8)
      .def(
          "print_info",
          [](const cast::CPUCostModel& self, int verbose) -> void {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect(std::cout);
            self.displayInfo({std::cout, verbose});
          },
          py::arg("verbose") = 1)
      .def(
          "show_entries",
          [](const cast::CPUCostModel& self, int nLines) -> void {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect(std::cout);
            self.showEntries(std::cout, nLines);
          },
          py::arg("n_lines") = 5)
      .def("clear_cache", &cast::CPUCostModel::clearCache)
      .def(
          "load_cache",
          [](cast::CPUCostModel& self, const cast::CPUPerformanceCache& cache) {
            if (auto e = self.loadCache(cache)) {
              throw std::runtime_error(
                  "Failed to load CPU performance cache: " +
                  llvm::toString(std::move(e)));
            }
          });
}

void bind_CPUOptimizer(py::module_& m) {
  // C++ inheritance is
  // CPUOptimizer -> Optimizer<CPUOptimizer> -> OptimizerBase
  // We bound OptimizerBase in bind_core.cpp, exposed run_circuit and
  // run_circuitgraph.
  // Now, enable_fusion, enable_canonicalization, enable_cfo, and
  // set_sizeonly_fusion_config must be bound here.
  py::class_<cast::CPUOptimizer, cast::OptimizerBase>(m, "CPUOptimizer")
      .def(py::init<>())
      .def(
          "print_info",
          [](const cast::CPUOptimizer& self, int verbose) -> void {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect(std::cout);
            self.displayInfo({std::cout, verbose});
          },
          py::arg("verbose") = 1)
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
           py::arg("size"));
}

} // end of anonymous namespace

PYBIND11_MODULE(pybind_cast_cpu, m) {
  m.doc() = "CAST CPU-related module";

  bind_simdWidth(m);
  bind_CPUStatevector<cast::CPUStatevectorFP32>(m, "CPUStatevectorFP32");
  bind_CPUStatevector<cast::CPUStatevectorFP64>(m, "CPUStatevectorFP64");
  bind_CPUKernelManager(m);
  bind_CPUPerformanceCache(m);
  bind_CPUCostModel(m);
  bind_CPUOptimizer(m);
}