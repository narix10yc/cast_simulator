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

  /// @brief The operation count. This function is not fully implemented yet.
  double opCount(double zeroTol = 1e-8) const;

  GateMatrixPtr gateMatrix() { return _gateMatrix; }
  const GateMatrixPtr& gateMatrix() const { return _gateMatrix; }

  NoiseChannelPtr noiseChannel() { return _noiseChannel; }
  const NoiseChannelPtr& noiseChannel() const { return _noiseChannel; }

  TargetQubitsType& qubits() { return _qubits; }
  const TargetQubitsType& qubits() const { return _qubits; }

  std::ostream& displayInfo(std::ostream& os, int verbose) const;

  static QuantumGatePtr Create(GateMatrixPtr gateMatrix,
                               NoiseChannelPtr noiseChannel,
                               const TargetQubitsType& qubits) {
    return std::make_shared<QuantumGate>(gateMatrix, noiseChannel, qubits);
  }
}; // class QuantumGate

QuantumGatePtr matmul(const QuantumGate& gateA, const QuantumGate& gateB);

}; // namespace cast

#endif // CAST_QUANTUM_GATE_H