#include "cast/internal/KrausRep.h"
#include "cast/internal/ChoiRep.h"

using namespace cast;

std::ostream& KrausRep::display(std::ostream& os) const {
  os << "KrausRep: " << _nQubits << " qubit(s) with "
     << nKraus() << " Kraus operator(s).\n";
  for (size_t i = 0; i < nKraus(); ++i) {
    os << "Kraus operator " << i << ":\n";
    _matrices[i].matrix().print(os);
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
      choiMatrix.real(r, c) =   krausReal[r] * krausReal[c] +
                                krausImag[r] * krausImag[c];
      choiMatrix.imag(r, c) = - krausReal[r] * krausImag[c] +
                                krausImag[r] * krausReal[c];
    }
  }

  for (size_t i = 1; i < nKraus; ++i) {
    krausReal = krausRep[i].matrix().reData();
    krausImag = krausRep[i].matrix().imData();
    for (size_t r = 0; r < edgeSize; ++r) {
      for (size_t c = 0; c < edgeSize; ++c) {
        choiMatrix.real(r, c) =   krausReal[r] * krausReal[c] +
                                  krausImag[r] * krausImag[c];
        choiMatrix.imag(r, c) = - krausReal[r] * krausImag[c] +
                                  krausImag[r] * krausReal[c];
      }
    }
  }

  return choiRep;
}