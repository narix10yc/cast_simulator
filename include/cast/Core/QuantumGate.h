#ifndef CAST_CORE_QUANTUM_GATE_H
#define CAST_CORE_QUANTUM_GATE_H

#include "cast/ADT/GateMatrix.h"
#include "cast/ADT/NoiseChannel.h"
#include <vector>
#include <cassert>

namespace cast {

class QuantumGate;
using QuantumGatePtr = std::shared_ptr<QuantumGate>;

class StandardQuantumGate;
using StandardQuantumGatePtr = std::shared_ptr<StandardQuantumGate>;

class SuperopQuantumGate;
using SuperopQuantumGatePtr = std::shared_ptr<SuperopQuantumGate>;

/// Recommended to use QuantumGate::Create() to create a QuantumGatePtr (which
/// is a shared_ptr<QuantumGate>).
class QuantumGate {
public:
  enum QuantumGateKind {
    QG_Base,
    QG_Standard, // Standard quantum gate
    QG_Superop,  // Superoperator quantum gate
    QG_End,
  };
private:
  QuantumGateKind _kind;
public:
  QuantumGateKind kind() const { return _kind; }
  using TargetQubitsType = std::vector<int>;
protected:
  // qubits are sorted in ascending order
  TargetQubitsType _qubits;
public:
  QuantumGate(QuantumGateKind kind) : _kind(kind) {}
  virtual ~QuantumGate() = default;

  int nQubits() const { return _qubits.size(); }

  /// @brief The operation count.
  virtual double opCount(double zeroTol) const {
    assert(false && "Calling from base class");
    return 0.0;
  }

  TargetQubitsType& qubits() { return _qubits; }
  const TargetQubitsType& qubits() const { return _qubits; }

  virtual std::ostream& displayInfo(std::ostream& os, int verbose) const {
    return os << "QuantumGate::displayInfo() not implemented";
  }

}; // class QuantumGate

QuantumGatePtr matmul(const QuantumGate* gateA, const QuantumGate* gateB);

/// @brief StandardQuantumGate consists of a GateMatrix and a NoiseChannel.
class StandardQuantumGate : public QuantumGate {
private:
  GateMatrixPtr _gateMatrix;
  NoiseChannelPtr _noiseChannel;
public:
  StandardQuantumGate(GateMatrixPtr gateMatrix,
                      NoiseChannelPtr noiseChannel,
                      const TargetQubitsType& qubits);

  GateMatrixPtr gateMatrix() { return _gateMatrix; }
  const GateMatrixPtr& gateMatrix() const { return _gateMatrix; }

  NoiseChannelPtr noiseChannel() { return _noiseChannel; }
  const NoiseChannelPtr& noiseChannel() const { return _noiseChannel; }

  double opCount(double zeroTol) const override;

  std::ostream& displayInfo(std::ostream& os, int verbose) const override;

  static StandardQuantumGatePtr Create(GateMatrixPtr gateMatrix,
                                       NoiseChannelPtr noiseChannel,
                                       const TargetQubitsType& qubits) {
    return std::make_shared<StandardQuantumGate>(
      gateMatrix, noiseChannel, qubits);
  }

  static bool classof(const QuantumGate* qg) {
    return qg->kind() == QG_Standard;
  }
}; // class StandardQuantumGate

/// @brief SuperopQuantumGate represents a quantum gate in Superoperator form.
class SuperopQuantumGate : public QuantumGate {
public:

  static bool classof(const QuantumGate* qg) {
    return qg->kind() == QG_Superop;
  }
}; // class SuperopQuantumGate

}; // namespace cast

#endif // CAST_CORE_QUANTUM_GATE_H