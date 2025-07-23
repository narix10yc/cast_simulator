#ifndef CAST_KRAUS_REP_H
#define CAST_KRAUS_REP_H

/* === KrausRep.h - Kraus representation of quantum channels === */

#include "cast/ADT/GateMatrix.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

namespace cast {

class KrausRep {
private:
  int _nQubits;
  std::vector<ScalarGateMatrix> _ops;

public:
  KrausRep(int nQubits) : _nQubits(nQubits), _ops() { assert(nQubits > 0); }

  const ScalarGateMatrix& operator[](size_t idx) const {
    assert(idx < nKraus());
    return _ops[idx];
  }

  ScalarGateMatrix& operator[](size_t idx) {
    assert(idx < nKraus());
    return _ops[idx];
  }

  int nQubits() const { return _nQubits; }

  const std::vector<ScalarGateMatrix>& getOps() const { return _ops; }
  std::vector<ScalarGateMatrix>& getOps() { return _ops; }

  void addMatrix(const ScalarGateMatrix& matrix) { _ops.push_back(matrix); }

  void addMatrix(ScalarGateMatrix&& matrix) {
    _ops.push_back(std::move(matrix));
  }

  size_t nKraus() const { return _ops.size(); }

  std::ostream& display(std::ostream& os) const;

}; // class KrausFormat

} // namespace cast

#endif // CAST_KRAUS_REP_H