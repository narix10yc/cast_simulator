#ifndef CAST_ADT_GATE_MATRIX_H
#define CAST_ADT_GATE_MATRIX_H

#include "cast/ADT/ComplexSquareMatrix.h"
#include <memory>
#include <vector>

namespace cast {

class GateMatrix;
using GateMatrixPtr = std::shared_ptr<GateMatrix>;

class ScalarGateMatrix;
using ScalarGateMatrixPtr = std::shared_ptr<ScalarGateMatrix>;

class UnitaryPermGateMatrix;
using UnitaryPermGateMatrixPtr = std::shared_ptr<UnitaryPermGateMatrix>;

class ParametrizedGateMatrix;
using ParametrizedGateMatrixPtr = std::shared_ptr<ParametrizedGateMatrix>;

/// @brief \c GateMatrix is a base class for all gate matrices.
/// It knows the number of qubits, but not which qubits.
/// Gate matrices here are `not` always unitary.
class GateMatrix {
public:
  enum GateMatrixKind {
    GM_Base,
    GM_Scalar, // Scalar matrix
    GM_UnitaryPerm, // Unitary permutation matrix
    GM_Parametrized, // Parametrized matrix
    GM_End,
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

  virtual std::ostream& displayInfo(std::ostream& os, int verbose=1) const {
    assert(false && "Calling from base class");
    return os;
  }

  static bool classof(const GateMatrix* gm) {
    return gm->kind() >= GM_Base && gm->kind() <= GM_End;
  }
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

  std::ostream& displayInfo(std::ostream& os, int verbose=1) const override;

  static bool classof(const GateMatrix* gm) {
    return gm->kind() == GM_Scalar;
  }

  static ScalarGateMatrixPtr X() {
    return std::make_shared<ScalarGateMatrix>(ComplexSquareMatrix::X());
  }

  static ScalarGateMatrixPtr Y() {
    return std::make_shared<ScalarGateMatrix>(ComplexSquareMatrix::Y());
  }

  static ScalarGateMatrixPtr Z() {
    return std::make_shared<ScalarGateMatrix>(ComplexSquareMatrix::Z());
  }

  static ScalarGateMatrixPtr H() {
    return std::make_shared<ScalarGateMatrix>(ComplexSquareMatrix::H());
  }

  // [cos(theta / 2),                -exp(i * lambda) * sin(theta / 2),
  //  exp(i * phi) * sin(theta / 2), exp(i * (phi + lambda)) * cos(theta / 2)]
  static ScalarGateMatrixPtr U1q(double theta, double phi, double lambda);

  // [    cos(theta/2), -i*sin(theta/2),
  //   -i*sin(theta/2),    cos(theta/2) ]
  static ScalarGateMatrixPtr RX(double theta);

  // [cos(theta/2), -sin(theta/2),
  //  sin(theta/2),  cos(theta/2) ]
  static ScalarGateMatrixPtr RY(double theta);
  
  // [exp(-i * theta/2),                0,
  //                  0, exp(i * theta/2) ]
  static ScalarGateMatrixPtr RZ(double theta);

  static ScalarGateMatrixPtr CX() {
    return std::make_shared<ScalarGateMatrix>(ComplexSquareMatrix::CX());
  }

}; // class ScalarGateMatrix

class UnitaryPermGateMatrix : public GateMatrix {
public:
  UnitaryPermGateMatrix(int nQubits)
    : GateMatrix(GM_UnitaryPerm, nQubits) {}

  static bool classof(const GateMatrix* gm) {
    return gm->kind() == GM_UnitaryPerm;
  }
}; // class UnitaryPermGateMatrix

class ParametrizedGateMatrix : public GateMatrix {
public:
  ParametrizedGateMatrix(int nQubits)
    : GateMatrix(GM_Parametrized, nQubits) {}

  static bool classof(const GateMatrix* gm) {
    return gm->kind() == GM_Parametrized;
  }
}; // class ParametrizedGateMatrix

/* Permute */

GateMatrixPtr permute(GateMatrixPtr gm, const std::vector<int>& flags);

/* Arithmatic Operator Overloading */

inline ScalarGateMatrix operator+(
    const ScalarGateMatrix& lhs, const ScalarGateMatrix& rhs) {
  assert(lhs.nQubits() == rhs.nQubits());
  return ScalarGateMatrix(lhs.matrix() + rhs.matrix());
}

inline ScalarGateMatrix operator+(
    const ScalarGateMatrix& lhs, double c) {
  return ScalarGateMatrix(lhs.matrix() + c);
}

inline ScalarGateMatrix operator+(
    double lhs, const ScalarGateMatrix& rhs) {
  return ScalarGateMatrix(lhs + rhs.matrix());
}

inline ScalarGateMatrix operator+(
    const ScalarGateMatrix& lhs, std::complex<double> c) {
  return ScalarGateMatrix(lhs.matrix() + c);
}

inline ScalarGateMatrix operator+(
    std::complex<double> c, const ScalarGateMatrix& rhs) {
  return ScalarGateMatrix(c + rhs.matrix());
}

inline ScalarGateMatrix& operator+=(
    ScalarGateMatrix& lhs, const ScalarGateMatrix& rhs) {
  assert(lhs.nQubits() == rhs.nQubits());
  lhs.matrix() += rhs.matrix();
  return lhs;
}

inline ScalarGateMatrix& operator+=(
    ScalarGateMatrix& lhs, double rhs) {
  lhs.matrix() += rhs;
  return lhs;
}

inline ScalarGateMatrix& operator+=(
    ScalarGateMatrix& lhs, std::complex<double> c) {
  lhs.matrix() += c;
  return lhs;
}

inline ScalarGateMatrix operator*(
    const ScalarGateMatrix& lhs, double c) {
  return ScalarGateMatrix(lhs.matrix() * c);
}

inline ScalarGateMatrix operator*(
    double c, const ScalarGateMatrix& rhs) {
  return ScalarGateMatrix(c * rhs.matrix());
}

inline ScalarGateMatrix operator*(
    const ScalarGateMatrix& lhs, std::complex<double> c) {
  return ScalarGateMatrix(lhs.matrix() * c);
}

inline ScalarGateMatrix operator*(
    std::complex<double> c, const ScalarGateMatrix& rhs) {
  return ScalarGateMatrix(c * rhs.matrix());
}

inline ScalarGateMatrix& operator*=(
    ScalarGateMatrix& lhs, double c) {
  lhs.matrix() *= c;
  return lhs;
}

inline ScalarGateMatrix& operator*=(
    ScalarGateMatrix& lhs, std::complex<double> c) {
  lhs.matrix() *= c;
  return lhs;
}


}; // namespace cast

#endif // CAST_ADT_GATE_MATRIX_H