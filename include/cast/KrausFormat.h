#ifndef CAST_KRAUSFORMAT_H
#define CAST_KRAUSFORMAT_H

#include "cast/GateMatrix.h"

#include <cstdint>
#include <iostream>
#include <cassert>

namespace cast {

class KrausFormat {
private:
  int _nQubits;
  std::vector<GateMatrix> _gateMatrices;
  std::vector<double> _weights;

public:
  KrausFormat(int nQubits) : _nQubits(nQubits) {
    _gateMatrices.resize(1ULL << nQubits);
    _weights.resize(1ULL << nQubits);
  }

  KrausFormat(const KrausFormat&) = delete;
  KrausFormat(KrausFormat&&) = delete;
  KrausFormat& operator=(const KrausFormat&) = delete;
  KrausFormat& operator=(KrausFormat&&) = delete;

  int nQubits() const { return _nQubits; }

  void setKrausOperator(
      unsigned idx, const GateMatrix& gateMatrix, double weight) {
    assert(idx < (1ULL << _nQubits));
    _gateMatrices[idx] = gateMatrix;
    _weights[idx] = weight;
  }

  void setKrausOperator(unsigned idx, GateMatrix&& gateMatrix, double weight) {
    assert(idx < (1ULL << _nQubits));
    _gateMatrices[idx] = std::move(gateMatrix);
    _weights[idx] = weight;
  }
  
  std::ostream& display(std::ostream& os) const;


}; // class KrausFormat


} // namespace cast


#endif // CAST_KRAUSFORMAT_H