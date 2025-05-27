#ifndef CAST_KRAUS_REP_H
#define CAST_KRAUS_REP_H

/* === KrausRep.h - Kraus representation of quantum channels === */

#include "cast/ADT/GateMatrix.h"

#include <memory>
#include <vector>
#include <iostream>
#include <cassert>

namespace cast {

class KrausRep {
private:
  int _nQubits;
  std::vector<ScalarGateMatrix> _matrices;
public:

  KrausRep(int nQubits) : _nQubits(nQubits) {
    assert(nQubits > 0);
    _matrices.reserve(1ULL << (2 * nQubits));
  }

  const ScalarGateMatrix& operator[](size_t idx) const {
    assert(idx < nKraus());
    return _matrices[idx];
  }

  ScalarGateMatrix& operator[](size_t idx) {
    assert(idx < nKraus());
    return _matrices[idx];
  }

  int nQubits() const { return _nQubits; }

  const std::vector<ScalarGateMatrix>& matrices() const { return _matrices; }
  std::vector<ScalarGateMatrix>& matrices() { return _matrices; }

  void addMatrix(const ScalarGateMatrix& matrix) {
    _matrices.push_back(matrix);
  }
  
  void addMatrix(ScalarGateMatrix&& matrix) {
    _matrices.push_back(std::move(matrix));
  }

  size_t nKraus() const { return _matrices.size(); }
  
  std::ostream& display(std::ostream& os) const;

}; // class KrausFormat

} // namespace cast

#endif // CAST_KRAUS_REP_H