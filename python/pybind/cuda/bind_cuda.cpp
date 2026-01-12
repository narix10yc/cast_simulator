#include "cast/CUDA/CUDA.h"
#include "cast/CUDA/CUDAKernelManager.h"

#include <pybind11/complex.h>
#include <pybind11/iostream.h>
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
          "print_info",
          [](const cast::CUDAKernelGenConfig& self, int verbose) -> void {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect(std::cout);
            self.displayInfo({std::cout, verbose});
          },
          py::arg("verbose") = 1);

  py::class_<cast::CUDAKernelHandler>(m, "CudaKernelHandler")
      .def(
          "print_info",
          [](const cast::CUDAKernelHandler& self, int verbose) -> void {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect(std::cout);
            self.displayInfo({std::cout, verbose});
          },
          py::arg("verbose") = 1);

  py::class_<cast::CUDAKernelManager>(m, "CUDAKernelManager")
      .def(py::init<>())
      .def(
          "print_info",
          [](const cast::CUDAKernelManager& self, int verbose) -> void {
            py::gil_scoped_acquire gil;
            py::scoped_ostream_redirect redirect(std::cout);
            self.displayInfo({std::cout, verbose});
          },
          py::arg("verbose") = 1)
      .def(
          "gen_gate",
          [](cast::CUDAKernelManager& self,
             const cast::CUDAKernelGenConfig& config,
             const cast::QuantumGatePtr& gate,
             const std::string& func_name) -> cast::CUDAKernelHandler {
            auto eHandler = self.genGate(config, gate, func_name);
            if (!eHandler) {
              throw std::runtime_error("Failed to generate gate: " +
                                       llvm::toString(eHandler.takeError()));
            }
            return *eHandler;
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
      .def("sync_compilation", &cast::CUDAKernelManager::syncCompilation)
      .def("sync_kernel_execution",
           &cast::CUDAKernelManager::syncKernelExecution);
}

// void bind_cudaOptimizer(py::module_& m) {
//   py::class_<cast::CUDAOptimizer, cast::OptimizerBase>(m, "CUDAOptimizer")
//       .def(py::init<>())
//       .def(
//           "print_info",
//           [](const cast::CUDAOptimizer& self, int verbose) -> void {
//             py::gil_scoped_acquire gil;
//             py::scoped_ostream_redirect redirect(std::cout);
//             self.displayInfo({std::cout, verbose});
//           },
//           py::arg("verbose") = 1)
//       .def("enable_fusion",
//            &cast::CUDAOptimizer::enableFusion,
//            py::arg("enable") = true)
//       .def("enable_canonicalization",
//            &cast::CUDAOptimizer::enableCanonicalization,
//            py::arg("enable") = true)
//       .def("enable_cfo",
//            &cast::CUDAOptimizer::enableCFO,
//            py::arg("enable") = true)
//       .def("set_sizeonly_fusion_config",
//            &cast::CUDAOptimizer::setSizeOnlyFusionConfig,
//            py::arg("size"));
// }

} // anonymous namespace

PYBIND11_MODULE(pybind_cast_cuda, m) {
  m.doc() = "CAST CUDA-related module";

  bind_cudaStatevector<cast::CUDAStatevectorFP32>(m, "CUDAStatevectorFP32");
  bind_cudaStatevector<cast::CUDAStatevectorFP64>(m, "CUDAStatevectorFP64");
  bind_cudaKernelManager(m);
  // bind_cudaOptimizer(m);
}
