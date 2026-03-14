#ifndef CAST_CPU_CPU_DENSITY_MATRIX_H
#define CAST_CPU_CPU_DENSITY_MATRIX_H

#include "cast/CPU/CPUStatevector.h"
#include <cstring>

namespace cast {

// This is almost the same as CPUStatevector with twice the number of qubits
template <typename ScalarType> class CPUDensityMatrix {
  CPUStatevector<ScalarType> sv;

public:
  CPUDensityMatrix(int nQubits, int simd_s) : sv(2 * nQubits, simd_s) {}

  int nQubits() const { return sv.nQubits() / 2; }

  void initialize() { sv.initialize(); }

  void randomize(int nThreads = 1) { sv.randomize(nThreads); }

  void* data() { return sv.data(); }

  std::ostream& print(std::ostream& os) const {
    if (nQubits() > 3) {
      os << "Density matrix has more than 2 qubits, "
            "resort to printing statevector.\n";
      return sv.print(os);
    }
    unsigned edgeSize = 1 << nQubits();
    for (unsigned r = 0; r < edgeSize; ++r) {
      for (unsigned c = 0; c < edgeSize; ++c) {
        utils::print_complex(os, sv.amp(c * edgeSize + r));
        os << ", ";
      }
      os << "\n";
    }
    return os;
  }
}; // class CPUDensityMatrix

// extern template class CPUDensityMatrix<float>;
// extern template class CPUDensityMatrix<double>;

} // namespace cast

#endif // CAST_CPU_CPU_DENSITY_MATRIX_H