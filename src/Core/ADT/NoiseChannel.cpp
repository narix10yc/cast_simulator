#include "cast/ADT/NoiseChannel.h"
#include "utils/iocolor.h"

using namespace cast;

std::ostream& KrausRep::display(std::ostream& os) const {
  os << "KrausRep: " << _nQubits << " qubit(s) with " << nKraus()
     << " Kraus operator(s).\n";
  for (size_t i = 0; i < nKraus(); ++i) {
    os << "Kraus operator " << i << ":\n";
    _ops[i].matrix().print(os);
  }
  return os;
}

std::ostream& ChoiRep::display(std::ostream& os) const {
  os << "ChoiRep: " << _nQubits << " qubit(s). Rank is " << rawRank() << ".\n";
  matrix().print(os);
  return os;
}

ChoiRep ChoiRep::FromKrausRep(const KrausRep& krausRep) {
  int nKraus = krausRep.nKraus();
  int nQubits = krausRep.nQubits();

  ChoiRep choiRep(nKraus, nQubits);
  auto& choiMatrix = choiRep.matrix();
  auto edgeSize = choiMatrix.edgeSize();

  // The Choi matrix is a sum of outer products of the Kraus operators.
  assert(nKraus > 0);
  auto* krausReal = krausRep[0].matrix().reData();
  auto* krausImag = krausRep[0].matrix().imData();
  for (size_t r = 0; r < edgeSize; ++r) {
    for (size_t c = 0; c < edgeSize; ++c) {
      choiMatrix.real(r, c) =
          krausReal[r] * krausReal[c] + krausImag[r] * krausImag[c];
      choiMatrix.imag(r, c) =
          -krausReal[r] * krausImag[c] + krausImag[r] * krausReal[c];
    }
  }

  for (size_t i = 1; i < nKraus; ++i) {
    krausReal = krausRep[i].matrix().reData();
    krausImag = krausRep[i].matrix().imData();
    for (size_t r = 0; r < edgeSize; ++r) {
      for (size_t c = 0; c < edgeSize; ++c) {
        choiMatrix.real(r, c) =
            krausReal[r] * krausReal[c] + krausImag[r] * krausImag[c];
        choiMatrix.imag(r, c) =
            -krausReal[r] * krausImag[c] + krausImag[r] * krausReal[c];
      }
    }
  }
  return choiRep;
}

int NoiseChannel::getRank() const {
  if (reps.choiRep) {
    return reps.choiRep->rank();
  }
  return -1;
}

std::ostream& NoiseChannel::displayInfo(std::ostream& os, int verbose) const {
  os << CYAN("Info of NoiseChannel @ " << this << "\n");
  os << "- Kraus Rep: " << (reps.krausRep ? "Available" : "No") << "\n";
  os << "- Choi Rep:  " << (reps.choiRep ? "Available" : "No") << "\n";
  if (verbose > 1) {
    if (reps.krausRep)
      reps.krausRep->display(os);
    if (reps.choiRep)
      reps.choiRep->display(os);
  }

  return os;
}

int NoiseChannel::nQubits() const {
  if (reps.krausRep) {
    return reps.krausRep->nQubits();
  } else if (reps.choiRep) {
    return reps.choiRep->nQubits();
  }
  assert(false && "No representation available");
  return -1;
}

NoiseChannelPtr NoiseChannel::SymmetricPauliChannel(double p) {
  assert(p >= 0 && p <= 1);
  int nQubits = 1;
  auto krausRep = std::make_shared<KrausRep>(nQubits);
  auto xMatrix = ScalarGateMatrix::X();
  auto yMatrix = ScalarGateMatrix::Y();
  auto zMatrix = ScalarGateMatrix::Z();
  auto iMatrix = ScalarGateMatrix::I1();

  // Kraus operators for the symmetric Pauli channel
  krausRep->addMatrix(*xMatrix * std::sqrt(p / 3));
  krausRep->addMatrix(*yMatrix * std::sqrt(p / 3));
  krausRep->addMatrix(*zMatrix * std::sqrt(p / 3));
  krausRep->addMatrix(*iMatrix * std::sqrt(1 - p));

  return std::make_shared<NoiseChannel>(krausRep);
}