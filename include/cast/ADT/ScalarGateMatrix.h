#ifndef CAST_ADT_SCALAR_GATE_MATRIX_H
#define CAST_ADT_SCALAR_GATE_MATRIX_H

#include "cast/ADT/ComplexSquareMatrix.h"

namespace cast {

/// @brief \c ScalarGateMatrix is a wrapper around \c ComplexSquareMatrix whose
/// edgeSize is always a power-of-2. 
class ScalarGateMatrix {
private:
  int _nQubits;
  ComplexSquareMatrix _matrix;
public:
  ScalarGateMatrix(int nQubits)
    : _nQubits(nQubits), _matrix(1ULL << nQubits) {}

  ScalarGateMatrix(const ComplexSquareMatrix& matrix) : _matrix(matrix) {
    this->_nQubits = static_cast<int>(std::log2(matrix.edgeSize()));
    assert(_nQubits > 0 && 1ULL << _nQubits == matrix.edgeSize() &&
           "Matrix size must be a power of 2");
  }

  ScalarGateMatrix(ComplexSquareMatrix&& matrix) noexcept
    : _matrix(std::move(matrix)) {
    this->_nQubits = static_cast<int>(std::log2(matrix.edgeSize()));
    assert(_nQubits > 0 && 1ULL << _nQubits == matrix.edgeSize() &&
           "Matrix size must be a power of 2");
  }

  int nQubits() const { return _nQubits; }

  ComplexSquareMatrix& matrix() { return _matrix; }
  const ComplexSquareMatrix& matrix() const { return _matrix; }

  static ScalarGateMatrix X() {
    return ScalarGateMatrix(ComplexSquareMatrix::X());
  }

  static ScalarGateMatrix Y() {
    return ScalarGateMatrix(ComplexSquareMatrix::Y());
  }

  static ScalarGateMatrix Z() {
    return ScalarGateMatrix(ComplexSquareMatrix::Z());
  }

  static ScalarGateMatrix H() {
    return ScalarGateMatrix(ComplexSquareMatrix::H());
  }

}; // class ScalarGateMatrix

} // namespace cast

#endif // CAST_ADT_SCALAR_GATE_MATRIX_H