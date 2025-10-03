#include "cast/Core/QuantumGate.h"
#include "utils/PrintSpan.h"
#include "utils/iocolor.h"
#include "llvm/Support/Casting.h"

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

  _qubits.resize(nQubits);
  for (unsigned i = 0; i < nQubits; i++)
    _qubits[i] = qubits[indices[i]];

  _gateMatrix = cast::permute(gateMatrix, indices);
  _noiseChannel = cast::permute(noiseChannel, indices);
}

ScalarGateMatrixPtr StandardQuantumGate::getScalarGM() {
  if (_gateMatrix == nullptr)
    return nullptr;
  if (llvm::isa<ScalarGateMatrix>(_gateMatrix.get()))
    return std::static_pointer_cast<ScalarGateMatrix>(_gateMatrix);
  return nullptr;
}

ConstScalarGateMatrixPtr StandardQuantumGate::getScalarGM() const {
  if (_gateMatrix == nullptr)
    return nullptr;
  if (llvm::isa<ScalarGateMatrix>(_gateMatrix.get()))
    return std::static_pointer_cast<const ScalarGateMatrix>(_gateMatrix);
  return nullptr;
}

void StandardQuantumGate::displayInfo(utils::InfoLogger logger) const {
  logger.put("Standard Quantum Gate").put("Target Qubits", std::span(_qubits));
  if (_gateMatrix == nullptr) {
    logger.put("Gate Matrix", "None");
  } else {
    logger.put("Gate Matrix");
    _gateMatrix->displayInfo(logger.indent());
  }

  // noise channel
  logger.put("Noise Channel", _noiseChannel.get());
}
