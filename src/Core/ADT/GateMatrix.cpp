#include "cast/ADT/GateMatrix.h"
#include "llvm/Support/Casting.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

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