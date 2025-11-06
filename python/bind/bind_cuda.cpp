#include "cast/CUDA/CUDA.h"
#include "cast/CUDA/CUDAKernelManager.h"
#include <pybind11/cast.h>
#include <pybind11/complex.h>
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

  py::class_<cast::CUDAKernelManager>(m, "CUDAKernelManager")
      .def(py::init<>())
      .def(
          "get_info",
          [](const cast::CUDAKernelManager& self, int verbose) -> std::string {
            std::ostringstream oss;
            self.displayInfo({oss, verbose});
            return oss.str();
          },
          py::arg("verbose") = 1);
}

} // anonymous namespace

void bind_cuda(py::module_& m) {
  bind_cudaStatevector<cast::CUDAStatevectorFP32>(m, "CUDAStatevectorFP32");
  bind_cudaStatevector<cast::CUDAStatevectorFP64>(m, "CUDAStatevectorFP64");
  bind_cudaKernelManager(m);
}
