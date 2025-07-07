#include "cast/ADT/ComplexSquareMatrix.h"
#include "utils/utils.h"
#include "utils/PrintSpan.h"

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
   0, 0, 0, 1,
   0, 0, 1, 0,
   0, 1, 0, 0},
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
   0, 0, 1, 0,
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
  std::memset(m.data(), 0, m.sizeInBytes());
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
  std::memcpy(result.imData(), imData(), halfSize() * sizeof(double));
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
  const auto edgeSize = A.edgeSize();
  C = ComplexSquareMatrix(edgeSize);
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

// helper functions to compute inverse
namespace {

constexpr double EPS = 1e-10; // tolerance to check singularity

/// The result of LUPDecompose is stored in LU, which contains both L and U 
/// matrices such that PA = LU.
/// The permutation is stored in P, where P[i] is the index of the row that was
/// swapped with row i during the decomposition. The last element P[edgeSize]
/// contains the number of row swaps made during the decomposition, which can be
/// used to compute determinant later.
/// This is the Wikipedia version adapted to complex matrices for our needs.
/// https://en.wikipedia.org/wiki/LU_decomposition
bool LUPDecompose(const ComplexSquareMatrix& A,
                  ComplexSquareMatrix& LU,
                  int* P) {
  const auto edgeSize = A.edgeSize();
  LU = A; // copy A to LU

  // initialize P to be the identity permutation
  for (int i = 0; i <= edgeSize; ++i)
    P[i] = i;

  for (int i = 0; i < edgeSize; ++i) {
    double pivotMag = 0.0;
    int pivotRow = i;

    for (int row = i; row < edgeSize; row++) {
      double mag = std::abs(LU.rc(row, i));
      if (mag > pivotMag) { 
        pivotMag = mag;
        pivotRow = row;
      }
    }

    if (pivotMag < EPS) {
      return false; // failure, matrix is degenerate
    }

    if (pivotRow != i) {
      // apply partial pivoting
      std::swap(P[i], P[pivotRow]);
      // swap row i and row imax in LU
      auto tmp = std::make_unique<double[]>(edgeSize);
      const size_t memSize = edgeSize * sizeof(double);
      // copy real part
      std::memcpy(tmp.get(),
                  LU.reBegin() + i * edgeSize,
                  memSize);
      std::memcpy(LU.reBegin() + i * edgeSize,
                  LU.reBegin() + pivotRow * edgeSize,
                  memSize);
      std::memcpy(LU.reBegin() + pivotRow * edgeSize,
                  tmp.get(),
                  memSize);

      // copy imag part
      std::memcpy(tmp.get(),
                  LU.imBegin() + i * edgeSize,
                  memSize);
      std::memcpy(LU.imBegin() + i * edgeSize,
                  LU.imBegin() + pivotRow * edgeSize,
                  memSize);
      std::memcpy(LU.imBegin() + pivotRow * edgeSize,
                  tmp.get(), 
                  memSize);

      // counting pivots starting from N (for determinant)
      P[edgeSize]++;
    }

    const auto factor = 1.0 / LU.rc(i, i);
    for (int j = i + 1; j < edgeSize; ++j) {
      auto newValue = LU.rc(j, i) * factor;
      LU.setRC(j, i, newValue);
      for (int col = i + 1; col < edgeSize; ++col)
        LU.setRC(j, col, LU.rc(j, col) - newValue * LU.rc(i, col));
    }
  }

  return true; // success
}

void LUPInvert(const ComplexSquareMatrix& LU,
               const int* P,
               ComplexSquareMatrix& AInv) {
  const auto edgeSize = LU.edgeSize();
  AInv = ComplexSquareMatrix(edgeSize);

  for (int j = 0; j < edgeSize; j++) {
    for (int i = 0; i < edgeSize; i++) {
      AInv.setRC(i, j, (P[i] == j ? 1.0 : 0.0), 0.0);

      for (int k = 0; k < i; k++)
        AInv.setRC(i, j, AInv.rc(i, j) - LU.rc(i, k) * AInv.rc(k, j));
    }

    for (int i = edgeSize - 1; i >= 0; i--) {
      for (int k = i + 1; k < edgeSize; k++)
        AInv.setRC(i, j, AInv.rc(i, j) - LU.rc(i, k) * AInv.rc(k, j));
      AInv.setRC(i, j, AInv.rc(i, j) / LU.rc(i, i));
    }
  }
}

} // end of anonymous namespace

bool cast::matinv(const ComplexSquareMatrix& A, ComplexSquareMatrix& AInv) {
  auto P = std::make_unique<int[]>(A.edgeSize() + 1);
  ComplexSquareMatrix LU;
  if (!LUPDecompose(A, LU, P.get()))
    return false; // LUP decomposition failed
  
  LUPInvert(LU, P.get(), AInv);
  return true; // success
}
