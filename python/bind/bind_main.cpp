#include "pybind11/pybind11.h"

namespace py = pybind11;

void bind_core(py::module_& m);
void bind_cpu(py::module_& m);

PYBIND11_MODULE(cast_python, m) {
  m.doc() = "Python bindings for the cast library";
  bind_core(m);
  bind_cpu(m);
}