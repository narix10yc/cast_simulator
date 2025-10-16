#ifndef CAST_ADT_COMPLEX_SQUARE_MATRIX_H
#define CAST_ADT_COMPLEX_SQUARE_MATRIX_H

#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>

namespace cast {

/// @brief \c ComplexSquareMatrix implements basic arithmatics to complex square
/// matrices with double precision.
class ComplexSquareMatrix {
  // constexpr static size_t AlignSize = 64;
private:
  size_t edgeSize_;
  double* data_;

  void allocData() {
    assert(edgeSize_ > 0);
    data_ = new double[edgeSize_ * edgeSize_ * 2];
    assert(data_ != nullptr && "Allocation failed");
  }

  void freeData() {
    delete[] data_;
    data_ = nullptr;
  }

public:
  ComplexSquareMatrix() : edgeSize_(0), data_(nullptr) {}

  ComplexSquareMatrix(size_t edgeSize) {
    edgeSize_ = edgeSize;
    allocData();
  }

  explicit ComplexSquareMatrix(std::initializer_list<double> re,
                               std::initializer_list<double> im) {
    assert(re.size() == im.size());
    size_t s = static_cast<size_t>(std::sqrt(re.size()));
    assert(s * s == re.size() && "Size is not a perfect square");
    this->edgeSize_ = s;
    allocData();
    std::memcpy(reData(), re.begin(), re.size() * sizeof(double));
    std::memcpy(imData(), im.begin(), im.size() * sizeof(double));
  }

  ComplexSquareMatrix(const ComplexSquareMatrix& other) {
    edgeSize_ = other.edgeSize_;
    allocData();
    std::memcpy(data_, other.data_, sizeInBytes());
  }

  ComplexSquareMatrix(ComplexSquareMatrix&& other) noexcept
      : edgeSize_(other.edgeSize_), data_(other.data_) {
    other.data_ = nullptr;
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

  ~ComplexSquareMatrix() { freeData(); }

  size_t edgeSize() const { return edgeSize_; }

  // Get the size of the real (or imag) part alone.
  // Equals to edgeSize() * edgeSize()
  size_t halfSize() const { return edgeSize_ * edgeSize_; }

  // Size of both real and imag parts. Equals to 2 * edgeSize() * edgeSize().
  // For the size of real or imag parts each, use halfSize().
  size_t size() const { return 2ULL * edgeSize_ * edgeSize_; }

  size_t sizeInBytes() const {
    return sizeof(double) * 2ULL * edgeSize_ * edgeSize_;
  }

  void fillZeros() { std::fill(data_, data_ + size(), 0.0); }

  double* data() { return data_; }
  const double* data() const { return data_; }

  double* reData() { return data_; }
  double* reBegin() { return data_; }
  double* reEnd() { return data_ + edgeSize_ * edgeSize_; }

  double* imData() { return data_ + edgeSize_ * edgeSize_; }
  double* imBegin() { return data_ + edgeSize_ * edgeSize_; }
  double* imEnd() { return data_ + 2 * edgeSize_ * edgeSize_; }

  const double* reData() const { return data_; }
  const double* reBegin() const { return data_; }
  const double* reEnd() const { return data_ + edgeSize_ * edgeSize_; }

  const double* imData() const { return data_ + edgeSize_ * edgeSize_; }
  const double* imBegin() const { return data_ + edgeSize_ * edgeSize_; }
  const double* imEnd() const { return data_ + 2 * edgeSize_ * edgeSize_; }

  double& real(unsigned row, unsigned col) {
    assert(row < edgeSize_ && col < edgeSize_ &&
           "Row or column index out of bounds");
    return data_[row * edgeSize_ + col];
  }

  double real(unsigned row, unsigned col) const {
    assert(row < edgeSize_ && col < edgeSize_ &&
           "Row or column index out of bounds");
    return data_[row * edgeSize_ + col];
  }

  double& imag(unsigned row, unsigned col) {
    assert(row < edgeSize_ && col < edgeSize_ &&
           "Row or column index out of bounds");
    return data_[(row + edgeSize_) * edgeSize_ + col];
  }

  double imag(unsigned row, unsigned col) const {
    assert(row < edgeSize_ && col < edgeSize_ &&
           "Row or column index out of bounds");
    return data_[(row + edgeSize_) * edgeSize_ + col];
  }

  std::complex<double> rc(unsigned row, unsigned col) const {
    assert(row < edgeSize_ && col < edgeSize_ &&
           "Row or column index out of bounds");
    return {real(row, col), imag(row, col)};
  }

  void setRC(unsigned row, unsigned col, double re, double im) {
    assert(row < edgeSize_ && col < edgeSize_ &&
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
  static ComplexSquareMatrix S();
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
double maximum_norm(const ComplexSquareMatrix& A, const ComplexSquareMatrix& B);

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

inline static cast::ComplexSquareMatrix
operator+(double c, const cast::ComplexSquareMatrix& m) {
  return m.operator+(c);
}

inline static cast::ComplexSquareMatrix
operator+(std::complex<double> c, const cast::ComplexSquareMatrix& m) {
  return m.operator+(c);
}

inline static cast::ComplexSquareMatrix
operator*(double c, const cast::ComplexSquareMatrix& m) {
  return m.operator*(c);
}

inline static cast::ComplexSquareMatrix
operator*(std::complex<double> c, const cast::ComplexSquareMatrix& m) {
  return m.operator*(c);
}

#endif // CAST_ADT_COMPLEX_SQUARE_MATRIX_H