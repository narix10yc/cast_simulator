#include "cast/ADT/ComplexSquareMatrix.h"
#include "utils/utils.h"

#include <random>

using namespace cast;

// For code collapse
#pragma region Constant Matrices

static const ComplexSquareMatrix matX(
  // real part
  {0, 1, 1, 0},
  // imag part
  {0, 0, 0, 0}
);

static const ComplexSquareMatrix matY(
  // real part
  {0, 0, 0, 0},
  // imag part
  {0, -1, 1, 0}
);

static const ComplexSquareMatrix matZ(
  // real part
  {1, 0, 0, -1},
  // imag part
  {0, 0, 0, 0}
);

static const ComplexSquareMatrix matH(
  // real part
  {M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, -M_SQRT1_2},
  // imag part
  {0, 0, 0, 0}
);

static const ComplexSquareMatrix matCX(
  // real part
  {1, 0, 0, 0,
   0, 1, 0, 0,
   0, 0, 0, 1,
   0, 0, 1, 0},
  // imag part
  {0, 0, 0, 0,
   0, 0, 0, 0,
   0, 0, 0, 0,
   0, 0, 0, 0}
);

static const ComplexSquareMatrix matCZ(
  // real part
  {1, 0, 0, 0,
   0, 1, 0, 0,
   0, 0, -1, 0,
   0, 0, 0, -1},
  // imag part
  {0, 0, 0, 0,
   0, 0, 0, 0,
   0, 0, 0, 0,
   0, 0, 0, 0}
);

static const ComplexSquareMatrix matSWAP(
  // real part
  {1, 0, 0, 0,
   0, 0, 1, 0,
   0, 1, 0, 0,
   0, 0, 0, 1},
  // imag part
  {0, 0, 0, 0,
   0, 0, 0, 0,
   0, 0, 0, 0,
   0, 0, 0, 0}
);

ComplexSquareMatrix ComplexSquareMatrix::X() {
  return matX;
}

ComplexSquareMatrix ComplexSquareMatrix::Y() {
  return matY;
}

ComplexSquareMatrix ComplexSquareMatrix::Z() {
  return matZ;
}

ComplexSquareMatrix ComplexSquareMatrix::H() {
  return matH;
}

ComplexSquareMatrix ComplexSquareMatrix::CX() {
  return matCX;
}

ComplexSquareMatrix ComplexSquareMatrix::CZ() {
  return matCZ;
}

ComplexSquareMatrix ComplexSquareMatrix::SWAP() {
  return matSWAP;
}

#pragma endregion

ComplexSquareMatrix ComplexSquareMatrix::eye(size_t edgeSize) {
  ComplexSquareMatrix m(edgeSize);
  std::memset(m.imData(), 0, m.halfSize() * sizeof(double));
  for (size_t i = 0; i < edgeSize; ++i)
    m.reBegin()[i * edgeSize + i] = 1.0;
  return m;
}

/* Implementation of Arithmatics of ComplexSquareMatrix */

ComplexSquareMatrix
ComplexSquareMatrix::operator+(const ComplexSquareMatrix& other) const {
  assert(_edgeSize == other._edgeSize);
  ComplexSquareMatrix result(_edgeSize);
  for (size_t i = 0; i < size(); ++i)
    result._data[i] = _data[i] + other._data[i];
  return result;
}

ComplexSquareMatrix ComplexSquareMatrix::operator+(double c) const {
  ComplexSquareMatrix result(_edgeSize);
  std::memset(result.imData(), 0, halfSize() * sizeof(double));
  for (size_t i = 0; i < halfSize(); ++i)
    result.reBegin()[i] = reBegin()[i] + c;
  return result;
}

ComplexSquareMatrix
ComplexSquareMatrix::operator+(std::complex<double> c) const {
  ComplexSquareMatrix result(_edgeSize);
  for (size_t i = 0; i < halfSize(); ++i)
    result.reBegin()[i] = reBegin()[i] + c.real();
  for (size_t i = 0; i < halfSize(); ++i)
    result.imBegin()[i] = imBegin()[i] + c.imag();
  return result;
}

ComplexSquareMatrix&
ComplexSquareMatrix::operator+=(const ComplexSquareMatrix& other) {
  assert(_edgeSize == other._edgeSize);
  for (size_t i = 0; i < size(); ++i)
    _data[i] += other._data[i];
  return *this;
}

ComplexSquareMatrix& ComplexSquareMatrix::operator+=(double c) {
  for (size_t i = 0; i < halfSize(); ++i)
    reBegin()[i] += c;
  return *this;
}

ComplexSquareMatrix& ComplexSquareMatrix::operator+=(std::complex<double> c) {
  for (size_t i = 0; i < halfSize(); ++i)
    reBegin()[i] += c.real();
  for (size_t i = 0; i < halfSize(); ++i)
    imBegin()[i] += c.imag();
  return *this;
}

ComplexSquareMatrix ComplexSquareMatrix::operator*(double c) const {
  ComplexSquareMatrix result(_edgeSize);
  for (size_t i = 0; i < size(); ++i)
    result._data[i] = c * _data[i];
  return result;
}

ComplexSquareMatrix& ComplexSquareMatrix::operator*=(double c) {
  for (size_t i = 0; i < size(); ++i)
    _data[i] *= c;
  return *this;
}

ComplexSquareMatrix
ComplexSquareMatrix::operator*(std::complex<double> c) const {
  ComplexSquareMatrix result(_edgeSize);
  for (size_t i = 0; i < halfSize(); ++i)
    result.reBegin()[i] = reBegin()[i] * c.real() - imBegin()[i] * c.imag();
  for (size_t i = 0; i < halfSize(); ++i)
    result.imBegin()[i] = reBegin()[i] * c.imag() + imBegin()[i] * c.real();
  return result;
}

ComplexSquareMatrix& ComplexSquareMatrix::operator*=(std::complex<double> c) {
  for (size_t i = 0; i < halfSize(); ++i) {
    auto oldRe = reBegin()[i];
    auto oldIm = imBegin()[i];
    auto newRe = oldRe * c.real() - oldIm * c.imag();
    auto newIm = oldRe * c.imag() + oldIm * c.real();
    reBegin()[i] = newRe;
    imBegin()[i] = newIm;
  }
  return *this;
}

std::ostream& ComplexSquareMatrix::print(std::ostream& os) const {
  if (_edgeSize == 0)
    return os << "[]\n";
  if (_edgeSize == 1)
    return utils::print_complex(os << "[", rc(0, 0)) << "]\n";

  // first (edgeSize - 1) rows
  os << "[";
  for (unsigned r = 0; r < _edgeSize - 1; ++r) {
    for (unsigned c = 0; c < _edgeSize; ++c)
      utils::print_complex(os, rc(r, c)) << ", ";
    os << "\n ";
  }

  // last row
  for (unsigned c = 0; c < _edgeSize - 1; c++)
    utils::print_complex(os, rc(_edgeSize - 1, c)) << ", ";
  utils::print_complex(os, rc(_edgeSize - 1, _edgeSize - 1));
  return os << " ]\n";
}

double cast::maximum_norm(const ComplexSquareMatrix& A,
                          const ComplexSquareMatrix& B) {
  assert(A.edgeSize() == B.edgeSize());
  assert(A.size() == B.size());
  double maxNorm = 0.0;
  for (size_t i = 0; i < A.size(); ++i) {
    double diff = std::abs(A.data()[i] - B.data()[i]);
    if (diff > maxNorm)
      maxNorm = diff;
  }
  return maxNorm;
}

// TODO: This is a non-optimized implementation of matrix multiplication.
void cast::matmul(const cast::ComplexSquareMatrix& A,
                  const cast::ComplexSquareMatrix& B,
                  cast::ComplexSquareMatrix& C) {
  assert(A.edgeSize() == B.edgeSize() && "Input size mismatch");
  assert(A.edgeSize() == C.edgeSize() && "Output size mismatch");
  const size_t edgeSize = A.edgeSize();
  for (size_t i = 0; i < edgeSize; ++i) {
    for (size_t j = 0; j < edgeSize; ++j) {
      auto re = 0.0;
      auto im = 0.0;
      for (size_t k = 0; k < edgeSize; ++k) {
        re += A.real(i, k) * B.real(k, j) - A.imag(i, k) * B.imag(k, j);
        im += A.real(i, k) * B.imag(k, j) + A.imag(i, k) * B.real(k, j);
      }
      C.real(i, j) = re;
      C.imag(i, j) = im;
    }
  }
}

// helper function to generate a random unitary matrix
namespace {
  // a dagger dotted with b
  void inner_product(const double* aRe, const double* aIm,
                     const double* bRe, const double* bIm,
                     double* resultRe, double* resultIm,
                     size_t length) {
    *resultRe = 0.0;
    *resultIm = 0.0;
    for (size_t i = 0; i < length; ++i) {
      //   (aRe - aIm * i) * (bRe + bIm * i)
      // = (aRe * bRe + aIm * bIm) + (aRe * bIm - aIm * bRe) * i
      *resultRe += aRe[i] * bRe[i] + aIm[i] * bIm[i];
      *resultIm += aRe[i] * bIm[i] - aIm[i] * bRe[i];
    }
  }

  void normalize(double* re, double* im, size_t length) {
    double norm = 0.0;
    for (size_t i = 0; i < length; ++i)
      norm += re[i] * re[i];
    for (size_t i = 0; i < length; ++i)
      norm += im[i] * im[i];
    norm = std::sqrt(norm);
    if (norm == 0.0) return; // avoid division by zero
    double factor = 1.0 / norm;
    for (size_t i = 0; i < length; ++i)
      re[i] *= factor;
    for (size_t i = 0; i < length; i++)
      im[i] *= factor;
  }

} // anonymous namespace

ComplexSquareMatrix ComplexSquareMatrix::RandomUnitary(size_t edgeSize) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist(0.0, 1.0);
  ComplexSquareMatrix m(edgeSize);

  for (unsigned r = 0; r < edgeSize; ++r) {
    for (unsigned c = 0; c < edgeSize; ++c)
      m.setRC(r, c, dist(gen), dist(gen));

    // project
    for (unsigned rr = 0; rr < r; ++rr) {
      double coefRe, coefIm;
      inner_product(
        m.reBegin() +  r * edgeSize, m.imBegin() +  r * edgeSize,
        m.reBegin() + rr * edgeSize, m.imBegin() + rr * edgeSize,
        &coefRe, &coefIm, edgeSize);
      // subtract row r by coef * row rr
      for (unsigned cc = 0; cc < edgeSize; ++cc) {
        double newRe = m.real(r, cc) - 
          (coefRe * m.real(rr, cc) + coefIm * m.imag(rr, cc));
        double newIm = m.imag(r, cc) - 
          (coefRe * m.imag(rr, cc) - coefIm * m.real(rr, cc));
        m.setRC(r, cc, newRe, newIm);
      }
    }

    // normalize
    normalize(m.reBegin() + r * edgeSize, 
              m.imBegin() + r * edgeSize,
              edgeSize);
  }

  return m;
}

