#include "cast/Core/QuantumGate.h"
#include <llvm/Support/Casting.h>

using namespace cast;

StandardQuantumGate::StandardQuantumGate(GateMatrixPtr gateMatrix,
                                         NoiseChannelPtr noiseChannel,
                                         const TargetQubitsType& qubits)
    : QuantumGate(QG_Standard) {
  const auto nQubits = qubits.size();
  std::vector<int> indices(nQubits);
  for (unsigned i = 0; i < nQubits; i++)
    indices[i] = i;

  std::ranges::sort(indices,
                    [&qubits](int i, int j) { return qubits[i] < qubits[j]; });

  qubits_.resize(nQubits);
  for (unsigned i = 0; i < nQubits; i++)
    qubits_[i] = qubits[indices[i]];

  gateMatrix_ = cast::permute(gateMatrix, indices);
  noiseChannel_ = cast::permute(noiseChannel, indices);
}

ScalarGateMatrixPtr StandardQuantumGate::getScalarGM() {
  if (gateMatrix_ == nullptr)
    return nullptr;
  if (llvm::isa<ScalarGateMatrix>(gateMatrix_.get()))
    return std::static_pointer_cast<ScalarGateMatrix>(gateMatrix_);
  return nullptr;
}

ConstScalarGateMatrixPtr StandardQuantumGate::getScalarGM() const {
  if (gateMatrix_ == nullptr)
    return nullptr;
  if (llvm::isa<ScalarGateMatrix>(gateMatrix_.get()))
    return std::static_pointer_cast<const ScalarGateMatrix>(gateMatrix_);
  return nullptr;
}

void StandardQuantumGate::displayInfo(utils::InfoLogger logger) const {
  logger.put("Standard Quantum Gate").put("Target Qubits", std::span(qubits_));
  if (gateMatrix_ == nullptr) {
    logger.put("Gate Matrix", "None");
  } else {
    logger.put("Gate Matrix");
    gateMatrix_->displayInfo(logger.indent());
  }

  // noise channel
  logger.put("Noise Channel", noiseChannel_.get());
}
