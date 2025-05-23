#include "llvm/Support/Casting.h"

#include "cast/QuantumGate.h"
#include "utils/iocolor.h"
#include "utils/PrintSpan.h"

using namespace cast;

QuantumGate::QuantumGate(GateMatrixPtr gateMatrix,
                         NoiseChannelPtr noiseChannel,
                         const TargetQubitsType& qubits) {
  const auto nQubits = qubits.size();
  std::vector<int> indices(nQubits);
  for (unsigned i = 0; i < nQubits; i++)
    indices[i] = i;

  std::ranges::sort(indices,[&qubits](int i, int j) {
    return qubits[i] < qubits[j];
  });

  _qubits.resize(nQubits);
  for (unsigned i = 0; i < nQubits; i++)
    _qubits[i] = qubits[indices[i]];

  _gateMatrix = cast::permute(gateMatrix, indices);
  _noiseChannel = cast::permute(noiseChannel, indices);
}

std::ostream& QuantumGate::displayInfo(std::ostream& os, int verbose) const {
  os << BOLDCYAN("=== Info of QuantumGate @ " << this << " === ")
     << "(Verbose " << verbose << ")\n";
     
  os << CYAN("- Target Qubits: ");
  utils::printSpan(os, std::span<const int>(_qubits)) << "\n";

  // gate matrix
  os << CYAN("- gateMatrix: ");
  if (_gateMatrix == nullptr)
    os << "nullptr";
  else if (auto* sMat = llvm::dyn_cast<ScalarGateMatrix>(_gateMatrix.get()))
    os << "ScalarGateMatrix @ " << sMat;
  else if (auto* uMat = llvm::dyn_cast<UnitaryPermGateMatrix>(_gateMatrix.get()))
    os << "UnitaryPermGateMatrix @ " << uMat;
  else if (auto* pMat = llvm::dyn_cast<ParametrizedGateMatrix>(_gateMatrix.get()))
    os << "ParametrizedGateMatrix @ " << pMat;
  else
    assert(false && "Unknown GateMatrix type");
  os << "\n";

  if (verbose > 1 && _gateMatrix != nullptr) {
    if (auto* sMat = llvm::dyn_cast<ScalarGateMatrix>(_gateMatrix.get()))
      sMat->matrix().print(os);
  }

  // noise channel
  os << CYAN("- noiseChannel: ");
  if (_noiseChannel == nullptr)
    os << "nullptr";
  else
    os << "NoiseChannel @ " << _noiseChannel.get();
  os << "\n";
  if (verbose > 1 && _noiseChannel != nullptr) {
    _noiseChannel->displayInfo(os, verbose - 1);
  }

  os << BOLDCYAN("========== End ==========\n");
  return os;
}