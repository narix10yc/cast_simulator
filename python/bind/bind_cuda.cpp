#include "cast/CUDA/CUDA.h"
#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAOptimizer.h"

#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

template <typename SVType>
void bind_cudaStatevector(py::module_& m, const char* pyName) {
  py::class_<SVType>(m, pyName)
      .def(py::init<int, bool>(),
           py::arg("num_qubits"),
           py::arg("initialize") = true)
      .def("initialize", &SVType::initialize)
      .def("randomize", &SVType::randomize)
      .def("num_qubits", &SVType::nQubits)
      .def("probability", &SVType::prob, py::arg("qubit"))
      .def("norm", &SVType::norm)
      .def("get_device_ptr", &SVType::getDevicePtr)
      .def(
          "__getitem__",
          [](const SVType& self, size_t idx) {
            if (idx >= self.size())
              throw std::out_of_range("Index out of range");
            return self.amp(idx);
          },
          py::arg("idx"));
}

void bind_cudaKernelManager(py::module_& m) {
  py::enum_<cast::CUDAMatrixLoadMode>(m, "CUDAMatrixLoadMode")
      .value("UseMatImmValues", cast::CUDAMatrixLoadMode::UseMatImmValues)
      .value("LoadInDefaultMemSpace",
             cast::CUDAMatrixLoadMode::LoadInDefaultMemSpace)
      .value("LoadInConstMemSpace",
             cast::CUDAMatrixLoadMode::LoadInConstMemSpace);

  py::class_<cast::CUDAKernelGenConfig>(m, "CUDAKernelGenConfig")
      .def(py::init<cast::Precision>(), py::arg("precision"))
      .def_readwrite("zero_tol", &cast::CUDAKernelGenConfig::zeroTol)
      .def_readwrite("one_tol", &cast::CUDAKernelGenConfig::oneTol)
      .def_readwrite("matrix_load_mode",
                     &cast::CUDAKernelGenConfig::matrixLoadMode)
      .def(
          "get_info",
          [](const cast::CUDAKernelGenConfig& self,
             int verbose) -> std::string {
            std::ostringstream oss;
            self.displayInfo({oss, verbose});
            return oss.str();
          },
          py::arg("verbose") = 1);

  py::class_<cast::CUDAKernelInfo>(m, "CUDAKernelInfo")
      .def_readonly("gate", &cast::CUDAKernelInfo::gate)
      .def_readonly("precision", &cast::CUDAKernelInfo::precision)
      .def_property_readonly("has_ptx",
                             [](const cast::CUDAKernelInfo& self) {
                               return !self.ptxString.empty();
                             })
      .def_property_readonly("has_cubin",
                             [](const cast::CUDAKernelInfo& self) {
                               return !self.cubinData.empty();
                             })
      .def("get_ptx",
           [](const cast::CUDAKernelInfo& self) -> std::string {
             return self.ptxString;
           })
      .def(
          "get_info",
          [](const cast::CUDAKernelInfo& self, int verbose) {
            std::ostringstream oss;
            self.displayInfo({oss, verbose});
            return oss.str();
          },
          py::arg("verbose") = 1);

  py::class_<cast::CUDAKernelManager::ExecutionResult>(
      m, "CUDAKernelExecutionResult")
      .def_readonly("kernel_name",
                    &cast::CUDAKernelManager::ExecutionResult::kernelName)
      .def(
          "get_info",
          [](const cast::CUDAKernelManager::ExecutionResult& self,
             int verbose) {
            std::ostringstream oss;
            self.displayInfo({oss, verbose});
            return oss.str();
          },
          py::arg("verbose") = 1)
      .def("get_compile_time",
           &cast::CUDAKernelManager::ExecutionResult::getCompileTime)
      .def("get_kernel_time",
           &cast::CUDAKernelManager::ExecutionResult::getKernelTime);

  py::class_<cast::CUDAKernelManager>(m, "CUDAKernelManager")
      .def(py::init<>())
      .def(
          "get_info",
          [](const cast::CUDAKernelManager& self, int verbose) -> std::string {
            std::ostringstream oss;
            self.displayInfo({oss, verbose});
            return oss.str();
          },
          py::arg("verbose") = 1)
      .def(
          "gen_gate",
          [](cast::CUDAKernelManager& self,
             const cast::CUDAKernelGenConfig& config,
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
      .def("gen_graph_gates",
           [](cast::CUDAKernelManager& self,
              const cast::CUDAKernelGenConfig& config,
              const cast::ir::CircuitGraphNode& graph,
              const std::string& pool_name) {
             if (auto e = self.genGraphGates(config, graph, pool_name)) {
               throw std::runtime_error(
                   "Failed to generate CUDA gates from graph: " +
                   llvm::toString(std::move(e)));
             }
           })
      .def(
          "set_launch_config",
          [](cast::CUDAKernelManager& self,
             uint64_t device_ptr,
             int num_qubits,
             int block_size) {
            self.setLaunchConfig(device_ptr, num_qubits, block_size);
            if (!self.isLaunchConfigValid()) {
              throw std::runtime_error("Launch configuration is not valid");
            }
          },
          py::arg("device_ptr"),
          py::arg("num_qubits"),
          py::arg("block_size") = 64)
      .def(
          "get_kernels_in_pool",
          [](const cast::CUDAKernelManager& self,
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
          "launch_kernel_fp32",
          [](cast::CUDAKernelManager& self,
             cast::CUDAStatevectorFP32& sv,
             cast::CUDAKernelInfo& kernel) {
            if (kernel.precision != cast::Precision::FP32) {
              throw std::runtime_error(
                  "Kernel precision mismatch: expected 32, got " +
                  std::to_string(static_cast<int>(kernel.precision)));
            }
            if (self.isLaunchConfigValid() == false) {
              throw std::runtime_error("Launch configuration is not valid. "
                                       "Call set_launch_config first.");
            }
            return self.enqueueKernelLaunch(kernel);
          },
          py::arg("sv"),
          py::arg("kernel"),
          py::return_value_policy::reference_internal);
}

void bind_cudaOptimizer(py::module_& m) {
  py::class_<cast::CUDAOptimizer, cast::OptimizerBase>(m, "CUDAOptimizer")
      .def(py::init<>())
      .def(
          "get_info",
          [](const cast::CUDAOptimizer& self, int verbose) {
            std::ostringstream oss;
            self.displayInfo({oss, verbose});
            return oss.str();
          },
          py::arg("verbose") = 1)
      .def("enable_fusion",
           &cast::CUDAOptimizer::enableFusion,
           py::arg("enable") = true)
      .def("enable_canonicalization",
           &cast::CUDAOptimizer::enableCanonicalization,
           py::arg("enable") = true)
      .def("enable_cfo",
           &cast::CUDAOptimizer::enableCFO,
           py::arg("enable") = true)
      .def("set_sizeonly_fusion_config",
           &cast::CUDAOptimizer::setSizeOnlyFusionConfig,
           py::arg("size"));
}

} // anonymous namespace

void bind_cuda(py::module_& m) {
  bind_cudaStatevector<cast::CUDAStatevectorFP32>(m, "CUDAStatevectorFP32");
  bind_cudaStatevector<cast::CUDAStatevectorFP64>(m, "CUDAStatevectorFP64");
  bind_cudaKernelManager(m);
  bind_cudaOptimizer(m);
}
