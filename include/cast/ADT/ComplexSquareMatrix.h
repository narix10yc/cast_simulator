#ifndef CAST_ADT_COMPLEX_SQUARE_MATRIX_H
#define CAST_ADT_COMPLEX_SQUARE_MATRIX_H

#include <complex>
#include <cassert>
#include <cstring>
#include <cmath>

namespace cast {

/// @brief \c ComplexSquareMatrix implements basic arithmatics to complex square
/// matrices with double precision. 
class ComplexSquareMatrix {
  // constexpr static size_t AlignSize = 64;
private:
  size_t _edgeSize;
  double* _data;

  void allocData() {
    assert(_edgeSize > 0);
    _data = new double[_edgeSize * _edgeSize * 2];
    assert(_data != nullptr && "Allocation failed");
  }

  void freeData() {
    delete[] _data;
    _data = nullptr;
  }
public:
  ComplexSquareMatrix() : _edgeSize(0), _data(nullptr) {}

  ComplexSquareMatrix(size_t edgeSize) {
    _edgeSize = edgeSize;
    allocData();
  }

  explicit ComplexSquareMatrix(std::initializer_list<double> re,
                               std::initializer_list<double> im) {
    assert(re.size() == im.size());
    size_t s = static_cast<size_t>(std::sqrt(re.size()));
    assert(s * s == re.size() && "Size is not a perfect square");
    this->_edgeSize = s;
    allocData();
    std::memcpy(reData(), re.begin(), re.size() * sizeof(double));
    std::memcpy(imData(), im.begin(), im.size() * sizeof(double));
  }

  ComplexSquareMatrix(const ComplexSquareMatrix& other) {
    _edgeSize = other._edgeSize;
    allocData();
    std::memcpy(_data, other._data, sizeInBytes());
  }

  ComplexSquareMatrix(ComplexSquareMatrix&& other) noexcept
    : _edgeSize(other._edgeSize), _data(other._data) {
    other._data = nullptr;
  }

  ComplexSquareMatrix& operator=(const ComplexSquareMatrix& other) {
    if (this == &other)
      return *this;
    this->~ComplexSquareMatrix();
    new (this) ComplexSquareMatrix(other);
    return *this;
  }

  ComplexSquareMatrix& operator=(ComplexSquareMatrix&& other) noexcept {
    if (this == &other)
      return *this;
    this->~ComplexSquareMatrix();
    new (this) ComplexSquareMatrix(std::move(other));
    return *this;
  }

  ~ComplexSquareMatrix() {
    freeData();
  }

  size_t edgeSize() const { return _edgeSize; }

  // Get the size of the real (or imag) part alone.
  // Equals to edgeSize() * edgeSize()
  size_t halfSize() const {
    return _edgeSize * _edgeSize;
  }

  // Size of both real and imag parts. Equals to 2 * edgeSize() * edgeSize().
  // For the size of real or imag parts each, use halfSize().
  size_t size() const {
    return 2ULL * _edgeSize * _edgeSize;
  }

  size_t sizeInBytes() const {
    return sizeof(double) * 2ULL * _edgeSize * _edgeSize;
  }

  void fillZeros() {
    std::fill(_data, _data + size(), 0.0);
  }

  double* data() { return _data; }
  const double* data() const { return _data; }

  double* reData() { return _data; }
  double* reBegin() { return _data; }
  double* reEnd() { return _data + _edgeSize * _edgeSize; }

  double* imData() { return _data + _edgeSize * _edgeSize; }
  double* imBegin() { return _data + _edgeSize * _edgeSize; }
  double* imEnd() { return _data + 2 * _edgeSize * _edgeSize; }
  
  const double* reData() const { return _data; }
  const double* reBegin() const { return _data; }
  const double* reEnd() const { return _data + _edgeSize * _edgeSize; }

  const double* imData() const { return _data + _edgeSize * _edgeSize; }
  const double* imBegin() const { return _data + _edgeSize * _edgeSize; }
  const double* imEnd() const { return _data + 2 * _edgeSize * _edgeSize; }

  double& real(unsigned row, unsigned col) {
    assert(row < _edgeSize && col < _edgeSize &&
           "Row or column index out of bounds");
    return _data[row * _edgeSize + col];
  }

  double real(unsigned row, unsigned col) const {
    assert(row < _edgeSize && col < _edgeSize &&
           "Row or column index out of bounds");
    return _data[row * _edgeSize + col];
  }

  double& imag(unsigned row, unsigned col) {
    assert(row < _edgeSize && col < _edgeSize &&
           "Row or column index out of bounds");
    return _data[(row + _edgeSize) * _edgeSize + col];
  }

  double imag(unsigned row, unsigned col) const {
    assert(row < _edgeSize && col < _edgeSize &&
           "Row or column index out of bounds");
    return _data[(row + _edgeSize) * _edgeSize + col];
  }

  std::complex<double> rc(unsigned row, unsigned col) const {
    assert(row < _edgeSize && col < _edgeSize &&
           "Row or column index out of bounds");
    return { real(row, col), imag(row, col) };
  }

  void setRC(unsigned row, unsigned col, double re, double im) {
    assert(row < _edgeSize && col < _edgeSize &&
           "Row or column index out of bounds");
    real(row, col) = re;
    imag(row, col) = im;
  }

  void setRC(unsigned row, unsigned col, std::complex<double> c) {
    return setRC(row, col, c.real(), c.imag());
  }

  /* Addition */

  ComplexSquareMatrix operator+(const ComplexSquareMatrix& other) const;
  ComplexSquareMatrix operator+(double c) const;
  ComplexSquareMatrix operator+(std::complex<double> c) const;

  ComplexSquareMatrix& operator+=(const ComplexSquareMatrix& other);
  ComplexSquareMatrix& operator+=(double c);
  ComplexSquareMatrix& operator+=(std::complex<double> c);

  /* Scalar-Matrix Multiplication */

  ComplexSquareMatrix operator*(double c) const;
  ComplexSquareMatrix operator*(std::complex<double> c) const;

  ComplexSquareMatrix& operator*=(double c);
  ComplexSquareMatrix& operator*=(std::complex<double> c);

  std::ostream& print(std::ostream& os) const;

  // Generate a random unitary matrix of the given edge size.
  static ComplexSquareMatrix RandomUnitary(size_t edgeSize);

  static ComplexSquareMatrix X();
  static ComplexSquareMatrix Y();
  static ComplexSquareMatrix Z();
  static ComplexSquareMatrix H();
  static ComplexSquareMatrix eye(size_t edgeSize);

  static ComplexSquareMatrix I1() { return eye(2); }
  static ComplexSquareMatrix I2() { return eye(4); }

  static ComplexSquareMatrix CX();
  static ComplexSquareMatrix CNOT() { return CX(); }
  static ComplexSquareMatrix CZ();
  static ComplexSquareMatrix SWAP();
  static ComplexSquareMatrix CP(double phi);
}; // class ComplexSquareMatrix

/// @brief The maximum norm of two matrices is defined as the maximum of the
/// absolute values of the entries in A - B. That is,
/// maximum_norm(A, B) = max_{i,j} |A_ij - B_ij|. 
double maximum_norm(const ComplexSquareMatrix& A,
                    const ComplexSquareMatrix& B);

/* Matrix-Matrix Multiplication */
/// @brief Compute the matrix product C = AB.
/// A, B, and C must have the same edge size.
void matmul(const cast::ComplexSquareMatrix& A,
            const cast::ComplexSquareMatrix& B,
            cast::ComplexSquareMatrix& C);

[[nodiscard("matinv could return false, indicating failure")]]
bool matinv(const cast::ComplexSquareMatrix& A,
            cast::ComplexSquareMatrix& AInv);
}; // namespace cast


/* Overload left addition and left multiplication */

inline static cast::ComplexSquareMatrix operator+(
    double c, const cast::ComplexSquareMatrix& m) {
  return m.operator+(c);
}

inline static cast::ComplexSquareMatrix operator+(
    std::complex<double> c, const cast::ComplexSquareMatrix& m) {
  return m.operator+(c);
}

inline static cast::ComplexSquareMatrix operator*(
    double c, const cast::ComplexSquareMatrix& m) {
  return m.operator*(c);
}

inline static cast::ComplexSquareMatrix operator*(
    std::complex<double> c, const cast::ComplexSquareMatrix& m) {
  return m.operator*(c);
}

#endif // CAST_ADT_COMPLEX_SQUARE_MATRIX_H