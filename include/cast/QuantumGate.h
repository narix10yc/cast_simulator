#ifndef CAST_QUANTUM_GATE_H
#define CAST_QUANTUM_GATE_H

#include "cast/ADT/GateMatrix.h"
#include "cast/ADT/NoiseChannel.h"
#include <vector>
#include <cassert>

namespace cast {

class QuantumGate;
using QuantumGatePtr = std::shared_ptr<QuantumGate>;

/// Recommended to use QuantumGate::Create() to create a QuantumGatePtr (which
/// is a shared_ptr<QuantumGate>).
class QuantumGate {
public:
  using TargetQubitsType = std::vector<int>;
private:
  GateMatrixPtr _gateMatrix;
  NoiseChannelPtr _noiseChannel;
  // qubits are sorted in ascending order
  TargetQubitsType _qubits;
public:
  QuantumGate(GateMatrixPtr gateMatrix,
              NoiseChannelPtr noiseChannel,
              const TargetQubitsType& qubits);

  int nQubits() const { return _qubits.size(); }

  TargetQubitsType& qubits() { return _qubits; }
  const TargetQubitsType& qubits() const { return _qubits; }

  std::ostream& displayInfo(std::ostream& os, int verbose) const;

  static QuantumGatePtr Create(GateMatrixPtr gateMatrix,
                               NoiseChannelPtr noiseChannel,
                               const TargetQubitsType& qubits) {
    return std::make_shared<QuantumGate>(gateMatrix, noiseChannel, qubits);
  }


}; // class QuantumGate

}; // namespace cast

#endif // CAST_QUANTUM_GATE_H