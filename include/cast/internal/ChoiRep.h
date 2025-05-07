#ifndef CAST_INTERNAL_CHOI_REP_H
#define CAST_INTERNAL_CHOI_REP_H

/* === ChoiRep.h - Choi representation of quantum channels === */

#include "cast/ADT/ComplexSquareMatrix.h"

namespace cast {

class KrausRep; // Forward declaration

class ChoiRep {
private:
  // Set to -1 if the rank is not computed yet.
  int _rank;
  int _nQubits;
  ComplexSquareMatrix _matrix;
  ChoiRep(int rank, int nQubits)
    : _rank(rank), _nQubits(nQubits), _matrix(1ULL << (2 * nQubits)) {
    assert(rank > 0 && nQubits > 0);
  }
public:
  ChoiRep(int nQubits)
    : _rank(-1), _nQubits(nQubits), _matrix(1ULL << (2 * nQubits)) {
    assert(nQubits > 0);
  }

  ChoiRep(const ComplexSquareMatrix& matrix) : _matrix(matrix) {
    this->_nQubits = static_cast<int>(std::log2(matrix.edgeSize()));
    assert(_nQubits > 0 && 1ULL << (2 * _nQubits) == matrix.size() &&
           "Matrix size must be a power of 2");
  }

  ChoiRep(ComplexSquareMatrix&& matrix) noexcept
    : _matrix(std::move(matrix)) {
    this->_nQubits = static_cast<int>(std::log2(matrix.edgeSize()));
    assert(_nQubits > 0 && 1ULL << (2 * _nQubits) == matrix.size() &&
           "Matrix size must be a power of 2");
  }

  void computeRank(double tol);

  // Possibly returns a negative value if the rank is not computed yet.
  int rawRank() const { return _rank; }

  int rank() const { 
    assert(_rank >= 0 && "Rank is not computed yet. "
                         "Either use rawRank() or call computeRank() first.");
    return _rank;
  }

  /// Equals to the sum of the outer products of the Kraus operators.
  /// Rank is deduced by the number of Kraus operators.
  /// Defined in QuantumChannel.cpp
  static ChoiRep FromKrausRep(const KrausRep& krausRep);
  
  int nQubits() const { return _nQubits; }

  ComplexSquareMatrix& matrix() { return _matrix; }
  const ComplexSquareMatrix& matrix() const { return _matrix; }
  
  std::ostream& display(std::ostream& os) const;

}; // class ChoiRep
  
} // namespace cast



#endif // CAST_INTERNAL_CHOI_REP_H