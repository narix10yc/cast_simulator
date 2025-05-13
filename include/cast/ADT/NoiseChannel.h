#ifndef CAST_QUANTUM_CHANNEL_H
#define CAST_QUANTUM_CHANNEL_H

#include "cast/internal/KrausRep.h"
#include "cast/internal/ChoiRep.h"

namespace cast {

class NoiseChannel;
using NoiseChannelPtr = std::shared_ptr<NoiseChannel>;

/// It is suggested to not copy \c NoiseChannel s directly. Use 
/// \c std::shared_ptr instead.
class NoiseChannel {
public:
  struct Representations {
    std::shared_ptr<KrausRep> krausRep;
    std::shared_ptr<ChoiRep> choiRep;
  };
  Representations reps;

  NoiseChannel(std::shared_ptr<KrausRep> krausRep)
    : reps(std::move(krausRep), nullptr) {}

  NoiseChannel(std::shared_ptr<ChoiRep> choiRep)
    : reps(nullptr, std::move(choiRep)) {}

  std::ostream& displayInfo(std::ostream& os, int verbose=1) const;

  int nQubits() const;

  /// @brief Get the Choi rank of the channel.
  /// The choi rank is the rank of the associated Choi matrix, which also equals
  /// to the minimum number of Kraus operators.
  int getRank() const;

  static NoiseChannelPtr SymmetricPauliChannel(double p);

}; // class NoiseChannel

} // namespace cast
 
#endif // CAST_QUANTUM_CHANNEL_H