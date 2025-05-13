#ifndef CAST_QUANTUM_GATE_H
#define CAST_QUANTUM_GATE_H

#include "cast/ADT/NoiseChannel.h"
#include "cast/ADT/GateMatrix.h"

namespace cast {

class QuantumGate;
using QuantumGatePtr = std::shared_ptr<QuantumGate>;

/// @brief \c QuantumGate is a wrapper around \c GateMatrix and \c NoiseChannel.
/// Each of them can be nullptr.
/// Recommended to use \c QuantumGatePtr (which is a shared pointer to 
/// \c QuantumGate ). Use factory constructor \c QuantumGate::Create() to return
/// a \c QuantumGatePtr.
class QuantumGate {
  GateMatrixPtr gateMatrix;
  NoiseChannelPtr noiseChannel;
public:
  QuantumGate(GateMatrixPtr gateMatrix,
              NoiseChannelPtr noiseChannel)
    : gateMatrix(std::move(gateMatrix))
    , noiseChannel(std::move(noiseChannel)) {
    checkValid();
  }

  /* Factory constructors */

  static QuantumGatePtr Create(GateMatrixPtr gateMatrix,
                               NoiseChannelPtr noiseChannel) {
    return std::make_shared<QuantumGate>(std::move(gateMatrix),
                                         std::move(noiseChannel));
  }

  static QuantumGatePtr Create(GateMatrixPtr gateMatrix) {
    return std::make_shared<QuantumGate>(std::move(gateMatrix), nullptr);
  }

  static QuantumGatePtr Create(NoiseChannelPtr noiseChannel) {
    return std::make_shared<QuantumGate>(nullptr, std::move(noiseChannel));
  }

  /* End of factory constructors */

  std::ostream& displayInfo(std::ostream& os, int verbose=1) const;

  void checkValid() const {
    if (gateMatrix && noiseChannel) {
      assert(gateMatrix->nQubits() == noiseChannel->nQubits());
    }
  }

  int nQubits() const {
    return gateMatrix->nQubits();
  }

  GateMatrixPtr getGateMatrix() const {
    return gateMatrix;
  }
  
  NoiseChannelPtr getNoise() const {
    return noiseChannel;
  }

  void setGateMatrix(GateMatrixPtr gateMatrix) {
    this->gateMatrix = std::move(gateMatrix);
  }

  void setNoise(NoiseChannelPtr noise) {
    this->noiseChannel = std::move(noise);
  }

  void setNoiseSymmetricPauliChannel(double p) {
    this->noiseChannel = NoiseChannel::SymmetricPauliChannel(p);
  }

}; // class QuantumGate
  
} // namespace cast


#endif // CAST_QUANTUM_GATE_H