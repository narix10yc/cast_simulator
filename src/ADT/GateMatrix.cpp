#include "cast/ADT/GateMatrix.h"
#include "llvm/Support/Casting.h"
#include "utils/iocolor.h"

using namespace cast;

std::ostream& ScalarGateMatrix::displayInfo(std::ostream& os, int verbose) const {
  os << "ScalarGateMatrix @ " << this << "\n";
  os << "- nQubits: " << _nQubits << "\n";
  return os;
} // namespace cast