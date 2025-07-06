#ifndef CAST_CORE_QUANTUM_GATE_H
#define CAST_CORE_QUANTUM_GATE_H

#include "cast/ADT/GateMatrix.h"
#include "cast/ADT/NoiseChannel.h"
#include <vector>
#include <cassert>
#include <algorithm>

namespace cast {

class QuantumGate;
using QuantumGatePtr = std::shared_ptr<QuantumGate>;
using ConstQuantumGatePtr = std::shared_ptr<const QuantumGate>;

class StandardQuantumGate;
using StandardQuantumGatePtr = std::shared_ptr<StandardQuantumGate>;
using ConstStandardQuantumGatePtr = std::shared_ptr<const StandardQuantumGate>;

class SuperopQuantumGate;
using SuperopQuantumGatePtr = std::shared_ptr<SuperopQuantumGate>;
using ConstSuperopQuantumGatePtr = std::shared_ptr<const SuperopQuantumGate>;

/// Recommended to always use QuantumGatePtr than QuantumGate directly.
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
  explicit QuantumGate(QuantumGateKind kind) : _kind(kind) {}
  virtual ~QuantumGate() = default;

  int nQubits() const { return _qubits.size(); }

  TargetQubitsType& qubits() { return _qubits; }
  const TargetQubitsType& qubits() const { return _qubits; }

  /// @brief The operation count.
  virtual double opCount(double zeroTol) const {
    assert(false && "Calling from base class");
    return 0.0;
  }

  // The inverse of this quantum gate. Return nullptr if the inverse cannot be
  // found.
  virtual QuantumGatePtr inverse() const {
    return nullptr;
  }

  virtual SuperopQuantumGatePtr getSuperopGate() const {
    assert(false && "QuantumGate::getSuperopGate called from base class");
    return nullptr;
  }

  virtual std::ostream& displayInfo(std::ostream& os, int verbose) const {
    return os << "QuantumGate::displayInfo() not implemented";
  }

  virtual void dumpInfo() const {
    displayInfo(std::cerr, 3);
  }

}; // class QuantumGate

/// @brief StandardQuantumGate consists of a GateMatrix and a NoiseChannel.
/// GateMatrix could be parametrized.
/// We take the convention that noise comes `after` the gate operation. For 
/// example, if the noise channel has Kraus set {E_k}, and the gate is U, then
/// the composite channel has Kraus set {E_k U}.
/// TODO: do we want to support parametrized noise channel fusion?
class StandardQuantumGate : public QuantumGate {
private:
  GateMatrixPtr _gateMatrix;
  NoiseChannelPtr _noiseChannel;
public:
  StandardQuantumGate(GateMatrixPtr gateMatrix,
                      NoiseChannelPtr noiseChannel,
                      const TargetQubitsType& qubits);

  GateMatrixPtr gateMatrix() { return _gateMatrix; }
  ConstGateMatrixPtr gateMatrix() const { return _gateMatrix; }

  NoiseChannelPtr noiseChannel() { return _noiseChannel; }
  ConstNoiseChannelPtr noiseChannel() const { return _noiseChannel; }

  // Set the noise channel to a symmetric Pauli channel with probability p.
  void setNoiseSPC(double p) {
    _noiseChannel = NoiseChannel::SymmetricPauliChannel(p);
  }

  double opCount(double zeroTol) const override;

  QuantumGatePtr inverse() const override;

  // Try to cast the gate matrix to ScalarGateMatrix. Returns nullptr if
  // the casting is not possible.
  ScalarGateMatrixPtr getScalarGM();

  // Try to cast the gate matrix to ScalarGateMatrix. Returns nullptr if
  // the casting is not possible.
  ConstScalarGateMatrixPtr getScalarGM() const;

  SuperopQuantumGatePtr getSuperopGate() const override;

  std::ostream& displayInfo(std::ostream& os, int verbose) const override;

  static StandardQuantumGatePtr Create(GateMatrixPtr gateMatrix,
                                       NoiseChannelPtr noiseChannel,
                                       const TargetQubitsType& qubits) {
    return std::make_shared<StandardQuantumGate>(
      gateMatrix, noiseChannel, qubits);
  }

  template<typename... Ints>
  static StandardQuantumGatePtr RandomUnitary(Ints... qubits) {
    static_assert((std::is_integral_v<Ints> && ...));
    constexpr auto nQubits = sizeof...(Ints);
    TargetQubitsType qubitsCopy{qubits...};
    std::ranges::sort(qubitsCopy);
    return StandardQuantumGate::Create(
      ScalarGateMatrix::RandomUnitary(nQubits),
      nullptr, // No noise channel
      qubitsCopy);
  }

  // @brief RandomUnitary generates a random unitary gate on the specified 
  // qubits. Only gate matrix is set, and no noise channel is applied.
  static StandardQuantumGatePtr RandomUnitary(const TargetQubitsType& qubits) {
    auto qubitsCopy = qubits;
    std::ranges::sort(qubitsCopy);
    return StandardQuantumGate::Create(
      ScalarGateMatrix::RandomUnitary(qubitsCopy.size()),
      nullptr, // No noise channel
      qubitsCopy);
  }

  // Get a single-qubit identity gate on qubit q.
  static StandardQuantumGatePtr I1(int q) {
    return StandardQuantumGate::Create(ScalarGateMatrix::I1(), nullptr, {q});
  }

  // Get a single-qubit Hadamard gate on qubit q.
  static StandardQuantumGatePtr H(int q) {
    return StandardQuantumGate::Create(ScalarGateMatrix::H(), nullptr, {q});
  }

  static bool classof(const QuantumGate* qg) {
    return qg->kind() == QG_Standard;
  }
}; // class StandardQuantumGate

/// @brief SuperopQuantumGate represents a quantum gate in Superoperator form.
class SuperopQuantumGate : public QuantumGate {
private:
  ScalarGateMatrixPtr _superopMatrix;
public:
  SuperopQuantumGate(ScalarGateMatrixPtr superopMatrix,
                     const TargetQubitsType& qubits);

  ScalarGateMatrixPtr getMatrix() { return _superopMatrix; }
  ConstScalarGateMatrixPtr getMatrix() const { return _superopMatrix; }

  double opCount(double zeroTol) const override;

  static SuperopQuantumGatePtr Create(ScalarGateMatrixPtr superopMatrix,
                                      const TargetQubitsType& qubits) {
    return std::make_shared<SuperopQuantumGate>(superopMatrix, qubits);
  }

  // This method returns a copy of this superop gate
  SuperopQuantumGatePtr getSuperopGate() const override {
    return SuperopQuantumGate::Create(_superopMatrix, _qubits);
  }

  std::ostream& displayInfo(std::ostream& os, int verbose) const override;

  static bool classof(const QuantumGate* qg) {
    return qg->kind() == QG_Superop;
  }
}; // class SuperopQuantumGate

// Return gateA @ gateB. In the context of quantum gates, gateB is applied
// first.
QuantumGatePtr matmul(const QuantumGate* gateA, const QuantumGate* gateB);

bool isCommuting(const QuantumGate* gateA,
                 const QuantumGate* gateB,
                 double tol = 1e-8);

}; // namespace cast

#endif // CAST_CORE_QUANTUM_GATE_H