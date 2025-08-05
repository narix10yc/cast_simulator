#ifndef CAST_ADT_GATE_MATRIX_H
#define CAST_ADT_GATE_MATRIX_H

#include "cast/ADT/ComplexSquareMatrix.h"
#include <memory>
#include <vector>

namespace cast {

class GateMatrix;
using GateMatrixPtr = std::shared_ptr<GateMatrix>;
using ConstGateMatrixPtr = std::shared_ptr<const GateMatrix>;

class ScalarGateMatrix;
using ScalarGateMatrixPtr = std::shared_ptr<ScalarGateMatrix>;
using ConstScalarGateMatrixPtr = std::shared_ptr<const ScalarGateMatrix>;

class UnitaryPermGateMatrix;
using UnitaryPermGateMatrixPtr = std::shared_ptr<UnitaryPermGateMatrix>;
using ConstUnitaryPermGateMatrixPtr =
    std::shared_ptr<const UnitaryPermGateMatrix>;

class ParametrizedGateMatrix;
using ParametrizedGateMatrixPtr = std::shared_ptr<ParametrizedGateMatrix>;
using ConstParametrizedGateMatrixPtr =
    std::shared_ptr<const ParametrizedGateMatrix>;

/// @brief \c GateMatrix is a base class for all gate matrices.
/// It knows the number of qubits, but not which qubits.
/// Gate matrices here are `not` always unitary.
class GateMatrix {
public:
  enum GateMatrixKind {
    GM_Base,
    GM_Scalar,       // Scalar matrix
    GM_UnitaryPerm,  // Unitary permutation matrix
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

  /// Compute the gate matrix of a subsystem.
  /// @param mask A bitmask indicating which qubits to keep. For example, a
  /// mask of 0b011 means to keep the least significant 2 qubits by partial
  /// tracing away the more significant qubits.
  virtual GateMatrixPtr subsystem(uint32_t mask) const {
    assert(false && "Not implemented yet (called from base class)");
    return nullptr;
  }

  virtual std::ostream& displayInfo(std::ostream& os, int verbose = 1) const {
    assert(false && "Not implemented yet (called from base class)");
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
      : GateMatrix(GM_Scalar, std::log2(matrix.edgeSize())),
        _matrix(std::move(matrix)) {
    assert(_nQubits > 0 && 1ULL << _nQubits == matrix.edgeSize() &&
           "Matrix size must be a power of 2");
  }

  ScalarGateMatrix(const ScalarGateMatrix& other)
      : GateMatrix(GM_Scalar, other._nQubits), _matrix(other._matrix) {}

  ScalarGateMatrix(ScalarGateMatrix&& other) noexcept
      : GateMatrix(GM_Scalar, other._nQubits),
        _matrix(std::move(other._matrix)) {
    other._nQubits = 0;
  }

  ScalarGateMatrix& operator=(const ScalarGateMatrix& other) {
    if (this == &other)
      return *this;
    _nQubits = other._nQubits;
    _matrix = other._matrix;
    return *this;
  }

  ScalarGateMatrix& operator=(ScalarGateMatrix&& other) noexcept {
    if (this == &other)
      return *this;
    _nQubits = other._nQubits;
    _matrix = std::move(other._matrix);
    return *this;
  }

  ComplexSquareMatrix& matrix() { return _matrix; }
  const ComplexSquareMatrix& matrix() const { return _matrix; }

  void fillZeros() { _matrix.fillZeros(); }

  GateMatrixPtr subsystem(uint32_t mask) const override;

  std::ostream& displayInfo(std::ostream& os, int verbose = 1) const override;

  static bool classof(const GateMatrix* gm) { return gm->kind() == GM_Scalar; }

  static ScalarGateMatrixPtr RandomUnitary(int nQubits) {
    return std::make_shared<ScalarGateMatrix>(
        ComplexSquareMatrix::RandomUnitary(1ULL << nQubits));
  }

  static ScalarGateMatrixPtr I1() {
    return std::make_shared<ScalarGateMatrix>(ComplexSquareMatrix::I1());
  }

  static ScalarGateMatrixPtr I2() {
    return std::make_shared<ScalarGateMatrix>(ComplexSquareMatrix::I2());
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

  static ScalarGateMatrixPtr S() {
    return std::make_shared<ScalarGateMatrix>(ComplexSquareMatrix::S());
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

  static ScalarGateMatrixPtr CP(double phi) {
    return std::make_shared<ScalarGateMatrix>(ComplexSquareMatrix::CP(phi));
  }
}; // class ScalarGateMatrix

class UnitaryPermGateMatrix : public GateMatrix {
private:
  struct IndexPhasePair {
    size_t index;
    double phase;
  };

  // array of index-phase pairs. For unitary permutation matrices, there is only
  // one non-zero entry in each row and column. Each non-zero entry is in the
  // complex unit circle, i.e., has a normed phase in [-pi, pi). \c _data
  // stores where the non-zero entries are, and their phases. For example,
  // Matrix[r, c] =
  //  exp(i * _data[r].phase) if _data[r].index == c;
  //                        0 if _data[r].index != c.
  // The phase is a output of std::atan2(im, re), with range (-pi, pi].
  IndexPhasePair* _data;

public:
  explicit UnitaryPermGateMatrix(int nQubits)
      : GateMatrix(GM_UnitaryPerm, nQubits),
        _data(new IndexPhasePair[1ULL << nQubits]) {}

  UnitaryPermGateMatrix(const UnitaryPermGateMatrix& other)
      : GateMatrix(GM_UnitaryPerm, other._nQubits) {
    _data = new IndexPhasePair[1ULL << _nQubits];
    std::memcpy(
        _data, other._data, sizeof(IndexPhasePair) * (1ULL << _nQubits));
  }

  UnitaryPermGateMatrix(UnitaryPermGateMatrix&& other) noexcept
      : GateMatrix(GM_UnitaryPerm, other._nQubits), _data(other._data) {
    other._data = nullptr;
  }

  ~UnitaryPermGateMatrix() { delete[] _data; }

  UnitaryPermGateMatrix& operator=(const UnitaryPermGateMatrix& other) {
    if (this == &other)
      return *this;
    this->~UnitaryPermGateMatrix();
    new (this) UnitaryPermGateMatrix(other);
    return *this;
  }

  UnitaryPermGateMatrix& operator=(UnitaryPermGateMatrix&& other) noexcept {
    if (this == &other)
      return *this;
    this->~UnitaryPermGateMatrix();
    new (this) UnitaryPermGateMatrix(std::move(other));
    return *this;
  }

  IndexPhasePair* data() { return _data; }
  const IndexPhasePair* data() const { return _data; }

  static UnitaryPermGateMatrixPtr FromGateMatrix(const GateMatrix* gm,
                                                 double zeroTol);

  static bool classof(const GateMatrix* gm) {
    return gm->kind() == GM_UnitaryPerm;
  }
}; // class UnitaryPermGateMatrix

class ParametrizedGateMatrix : public GateMatrix {
public:
  ParametrizedGateMatrix(int nQubits) : GateMatrix(GM_Parametrized, nQubits) {}

  static bool classof(const GateMatrix* gm) {
    return gm->kind() == GM_Parametrized;
  }
}; // class ParametrizedGateMatrix

/* Permute. Implemented in src/Core/Permute.cpp */

/// @brief Permute the gate matrix according to the given flags. Flags are
/// specified such that newQubits[flags[i]] = oldQubits[i].
GateMatrixPtr permute(GateMatrixPtr gm, const std::vector<int>& flags);

/* Arithmatic Operator Overloading */

/* Addition */

inline ScalarGateMatrix& operator+=(ScalarGateMatrix& lhs,
                                    const ScalarGateMatrix& rhs) {
  assert(lhs.nQubits() == rhs.nQubits());
  lhs.matrix() += rhs.matrix();
  return lhs;
}

inline ScalarGateMatrix& operator+=(ScalarGateMatrix& lhs, double rhs) {
  lhs.matrix() += rhs;
  return lhs;
}

inline ScalarGateMatrix& operator+=(ScalarGateMatrix& lhs,
                                    std::complex<double> c) {
  lhs.matrix() += c;
  return lhs;
}

inline ScalarGateMatrix operator+(const ScalarGateMatrix& lhs,
                                  const ScalarGateMatrix& rhs) {
  assert(lhs.nQubits() == rhs.nQubits());
  return ScalarGateMatrix(lhs.matrix() + rhs.matrix());
}

inline ScalarGateMatrix operator+(const ScalarGateMatrix& lhs, double c) {
  auto copy = lhs;
  copy += c;
  return copy;
}

inline ScalarGateMatrix operator+(double c, const ScalarGateMatrix& rhs) {
  return rhs + c;
}

inline ScalarGateMatrix operator+(const ScalarGateMatrix& lhs,
                                  std::complex<double> c) {
  auto copy = lhs;
  copy += c;
  return copy;
}

inline ScalarGateMatrix operator+(std::complex<double> c,
                                  const ScalarGateMatrix& rhs) {
  return rhs + c;
}

/* Multiplication */

inline ScalarGateMatrix& operator*=(ScalarGateMatrix& lhs, double c) {
  lhs.matrix() *= c;
  return lhs;
}

inline ScalarGateMatrix& operator*=(ScalarGateMatrix& lhs,
                                    std::complex<double> c) {
  lhs.matrix() *= c;
  return lhs;
}

inline ScalarGateMatrix operator*(const ScalarGateMatrix& lhs, double c) {
  auto copy = lhs;
  copy *= c;
  return copy;
}

inline ScalarGateMatrix operator*(double c, const ScalarGateMatrix& rhs) {
  return rhs * c;
}

inline ScalarGateMatrix operator*(const ScalarGateMatrix& lhs,
                                  std::complex<double> c) {
  auto copy = lhs;
  copy *= c;
  return copy;
}

inline ScalarGateMatrix operator*(std::complex<double> c,
                                  const ScalarGateMatrix& rhs) {
  return rhs * c;
}

}; // namespace cast

#endif // CAST_ADT_GATE_MATRIX_H