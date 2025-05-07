#ifndef CAST_QUANTUM_CHANNEL_H
#define CAST_QUANTUM_CHANNEL_H

#include "cast/internal/KrausRep.h"
#include "cast/internal/ChoiRep.h"
#include "llvm/ADT/SmallVector.h"

namespace cast {

/// \brief QuantumChannel represents a quantum channel. It is suggested to not
/// copy \c QuantumChannel s directly. Use \c std::shared_ptr instead.
class QuantumChannel {
public:
  llvm::SmallVector<int> qubits;
  struct Representations {
    std::shared_ptr<KrausRep> krausRep;
    std::shared_ptr<ChoiRep> choiRep;
  };
  mutable Representations reps;

  QuantumChannel(const llvm::SmallVector<int>& qubits)
    : qubits(qubits), reps() {}

  std::ostream& displayInfo(std::ostream& os, int verbose=1) const;

  /// @brief Get the Choi rank of the channel.
  /// The choi rank is the rank of the associated Choi matrix, which also equals
  /// to the minimum number of Kraus operators.
  int getRank() const;

}; // class QuantumChannel

} // namespace cast

#endif // CAST_QUANTUM_CHANNEL_H