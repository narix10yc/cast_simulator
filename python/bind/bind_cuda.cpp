#include "cast/CUDA/CUDA.h"
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

void bind_cudaStatevector(py::module_& m) {
  py::class_<cast::CUDAStatevectorFP32>(m, "CUDAStatevectorFP32")
      .def(py::init<int, bool>(),
           py::arg("num_qubits"),
           py::arg("initialize") = true)
      .def("initialize", &cast::CUDAStatevectorFP32::initialize)
      .def("randomize", &cast::CUDAStatevectorFP32::randomize);
}

} // anonymous namespace

void bind_cuda(py::module_& m) { bind_cudaStatevector(m); }
