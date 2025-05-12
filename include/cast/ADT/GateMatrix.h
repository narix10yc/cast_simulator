#ifndef CAST_ADT_GATE_MATRIX_H
#define CAST_ADT_GATE_MATRIX_H

#include "cast/ADT/ComplexSquareMatrix.h"

namespace cast {

class GateMatrix {
public:
  enum GateMatrixKind {
    GM_Scalar, // Scalar matrix
    GM_UnitaryPerm, // Unitary permutation matrix
    GM_Parametrized, // Parametrized matrix
    GM_Unknown,
  };
protected:
  GateMatrixKind _kind;
  int _nQubits;
public:
  GateMatrix(GateMatrixKind kind, int nQubits) 
    : _kind(kind), _nQubits(nQubits) {}

  virtual ~GateMatrix() = default;

  GateMatrixKind kind() const { return _kind; }
  int nQubits() const { return _nQubits; }

}; // class GateMatrix

/// @brief \c ScalarGateMatrix is a wrapper around \c ComplexSquareMatrix whose
/// edgeSize is always a power-of-2. 
class ScalarGateMatrix : public GateMatrix {
private:
  ComplexSquareMatrix _matrix;
public:
  ScalarGateMatrix(int nQubits)
    : GateMatrix(GM_Scalar, nQubits), _matrix(1ULL << nQubits) {}

  ScalarGateMatrix(const ComplexSquareMatrix& matrix)
    : GateMatrix(GM_Scalar, std::log2(matrix.edgeSize())), _matrix(matrix) {
    assert(_nQubits > 0 && 1ULL << _nQubits == matrix.edgeSize() &&
           "Matrix size must be a power of 2");
  }

  ScalarGateMatrix(ComplexSquareMatrix&& matrix)
    : GateMatrix(GM_Scalar, std::log2(matrix.edgeSize()))
    , _matrix(std::move(matrix)) {
    assert(_nQubits > 0 && 1ULL << _nQubits == matrix.edgeSize() &&
           "Matrix size must be a power of 2");
  }

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

}; // namespace cast

#endif // CAST_ADT_GATE_MATRIX_H