#include "cast/ADT/PatternMatrix.h"
#include "cast/ADT/ComplexSquareMatrix.h"

using namespace cast;

PatternMatrixPtr PatternMatrix::FromComplexSquareMatrix(
    const ComplexSquareMatrix& matrix, double zTol, double oTol) {

  auto edgeSize = matrix.edgeSize();

  auto pMat = std::make_shared<PatternMatrix>(edgeSize);
  assert(pMat != nullptr);

  // real part
  for (size_t r = 0; r < edgeSize; ++r) {
    for (size_t c = 0; c < edgeSize; ++c) {
      auto re = matrix.real(r, c);
      if (std::abs(re) <= zTol)
        pMat->setRe(r, c, Pattern::Zero);
      else if (std::abs(re - 1.0) <= oTol)
        pMat->setRe(r, c, Pattern::PlusOne);
      else if (std::abs(re + 1.0) <= oTol)
        pMat->setRe(r, c, Pattern::MinusOne);
      else
        pMat->setRe(r, c, Pattern::Generic);
    }
  }

  // imag part
  for (size_t r = 0; r < edgeSize; ++r) {
    for (size_t c = 0; c < edgeSize; ++c) {
      auto im = matrix.imag(r, c);
      if (std::abs(im) <= zTol)
        pMat->setIm(r, c, Pattern::Zero);
      else if (std::abs(im - 1.0) <= oTol)
        pMat->setIm(r, c, Pattern::PlusOne);
      else if (std::abs(im + 1.0) <= oTol)
        pMat->setIm(r, c, Pattern::MinusOne);
      else
        pMat->setIm(r, c, Pattern::Generic);
    }
  }

  return pMat;
}