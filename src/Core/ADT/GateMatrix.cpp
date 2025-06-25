#include "cast/ADT/GateMatrix.h"
#include "llvm/Support/Casting.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

#include "llvm/Support/Casting.h"

using namespace cast;

std::ostream& ScalarGateMatrix::displayInfo(std::ostream& os, int verbose) const {
  os << "ScalarGateMatrix @ " << this << "\n";
  os << "- nQubits: " << _nQubits << "\n";
  return os;
} // namespace cast

ScalarGateMatrixPtr ScalarGateMatrix::U1q(
    double theta, double phi, double lambda) {
  auto matrixPtr = std::make_shared<ScalarGateMatrix>(1);
  auto& matrix = matrixPtr->matrix();

  matrix.real(0, 0) = std::cos(theta * 0.5);
  matrix.imag(0, 0) = 0.0;

  matrix.real(0, 1) = -std::cos(lambda) * std::sin(theta * 0.5);
  matrix.imag(0, 1) = -std::sin(lambda) * std::sin(theta * 0.5);

  matrix.real(1, 1) = std::cos(phi + lambda) * std::cos(theta * 0.5);
  matrix.imag(1, 0) = std::sin(phi) * std::sin(theta * 0.5);
  
  matrix.real(1, 0) = std::cos(phi) * std::sin(theta * 0.5);
  matrix.imag(1, 1) = std::sin(phi + lambda) * std::cos(theta * 0.5);
  return matrixPtr;
}

ScalarGateMatrixPtr ScalarGateMatrix::RX(double theta) {
  auto matrixPtr = std::make_shared<ScalarGateMatrix>(1);
  auto& matrix = matrixPtr->matrix();
  matrix.setRC(0, 0, std::cos(theta * 0.5), 0.0);
  matrix.setRC(0, 1, 0.0, -std::sin(theta * 0.5));
  matrix.setRC(1, 0, 0.0, -std::sin(theta * 0.5));
  matrix.setRC(1, 1, std::cos(theta * 0.5), 0.0);
  return matrixPtr;
}

ScalarGateMatrixPtr ScalarGateMatrix::RY(double theta) {
  auto matrixPtr = std::make_shared<ScalarGateMatrix>(1);
  auto& matrix = matrixPtr->matrix();
  matrix.setRC(0, 0, std::cos(theta * 0.5), 0.0);
  matrix.setRC(0, 1, -std::sin(theta * 0.5), 0.0);
  matrix.setRC(1, 0, std::sin(theta * 0.5), 0.0);
  matrix.setRC(1, 1, std::cos(theta * 0.5), 0.0);
  return matrixPtr;
}

ScalarGateMatrixPtr ScalarGateMatrix::RZ(double theta) {
  auto matrixPtr = std::make_shared<ScalarGateMatrix>(1);
  auto& matrix = matrixPtr->matrix();
  matrix.setRC(0, 0, std::cos(theta * 0.5), -std::sin(theta * 0.5));
  matrix.setRC(0, 1, 0.0, 0.0);
  matrix.setRC(1, 0, 0.0, 0.0);
  matrix.setRC(1, 1, std::cos(theta * 0.5), std::sin(theta * 0.5));
  return matrixPtr;
}

UnitaryPermGateMatrixPtr UnitaryPermGateMatrix::FromGateMatrix(
    const GateMatrix* gm, double zeroTol) {
  if (gm == nullptr)
    return nullptr;
  if (const auto* upGM = llvm::dyn_cast<UnitaryPermGateMatrix>(gm)) {
    // Create a copy of the UnitaryPermGateMatrix.
    return std::make_shared<UnitaryPermGateMatrix>(*upGM);
  }
  if (const auto* scalarGM = llvm::dyn_cast<ScalarGateMatrix>(gm)) {
    // Convert ScalarGateMatrix to UnitaryPermGateMatrix.
    auto upGM = std::make_shared<UnitaryPermGateMatrix>(scalarGM->nQubits());
    const auto& matrix = scalarGM->matrix();
    for (unsigned r = 0; r < matrix.edgeSize(); ++r) {
      bool hasNonZero = false;
      for (unsigned c = 0; c < matrix.edgeSize(); ++c) {
        const auto re = matrix.real(r, c);
        const auto im = matrix.imag(r, c);
        bool isZero = re * re + im * im < zeroTol * zeroTol;
        if (isZero) {
          // skip zero entries
          continue;
        }
        if (hasNonZero) { // !isZero && hasNonZero
          // More than one non-zero entries in this row,
          // not a unitary permutation.
          return nullptr;
        } 
        // !isZero && !hasNonZero
        hasNonZero = true;
        upGM->data()[r].index = c;
        upGM->data()[r].phase = std::atan2(im, re);
        continue;
      }
      if (!hasNonZero) {
        // This row is full of zeros. Cannot be a unitary permutation.
        return nullptr;
      }
    }
    return upGM;
  }
  assert(false &&
    "Unsupported GateMatrix type for conversion to UnitaryPermGateMatrix");
  return nullptr;
}

GateMatrixPtr ScalarGateMatrix::subsystem(uint32_t mask) const {
  assert(mask != 0 && "Mask must not be zero");
  assert(false && "Unimplemented");
  return nullptr;
}