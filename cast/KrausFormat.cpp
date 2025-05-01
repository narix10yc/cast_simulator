#include "cast/KrausFormat.h"

using namespace cast;

std::ostream& KrausFormat::display(std::ostream& os) const {
  for (int i = 0; i < (1ULL << _nQubits); ++i) {
    os << "weight " << _weights[i] << ":\n";
    _gateMatrices[i].printCMat(os) << "\n";
  }
  return os;
}