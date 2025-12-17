#include "cast/ADT/PatternMatrix.h"
#include "cast/ADT/ComplexSquareMatrix.h"

using namespace cast;

PatternMatrixPtr PatternMatrix::FromComplexSquareMatrix(
    const ComplexSquareMatrix& matrix, double zTol, double oTol) {

  auto pMat = std::make_shared<PatternMatrix>(matrix.edgeSize());
  assert(pMat != nullptr);

  // real part
  for (size_t i = 0; i < matrix.halfSize(); ++i) {
    auto re = matrix.reData()[i];
    if (std::abs(re) <= zTol)
      pMat->setEntry(i, Pattern::Zero);
    else if (std::abs(re - 1.0) <= oTol)
      pMat->setEntry(i, Pattern::PlusOne);
    else if (std::abs(re + 1.0) <= oTol)
      pMat->setEntry(i, Pattern::MinusOne);
    else
      pMat->setEntry(i, Pattern::Generic);
  }

  // imag part
  for (size_t i = 0; i < matrix.halfSize(); ++i) {
    auto im = matrix.imData()[i];
    if (std::abs(im) <= zTol)
      pMat->setEntry(i + matrix.halfSize(), Pattern::Zero);
    else if (std::abs(im - 1.0) <= oTol)
      pMat->setEntry(i + matrix.halfSize(), Pattern::PlusOne);
    else if (std::abs(im + 1.0) <= oTol)
      pMat->setEntry(i + matrix.halfSize(), Pattern::MinusOne);
    else
      pMat->setEntry(i + matrix.halfSize(), Pattern::Generic);
  }

  return pMat;
}