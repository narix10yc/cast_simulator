#include "cast/Core/QuantumGate.h"
#include "utils/iocolor.h"
#include "utils/PrintSpan.h"
#include "llvm/Support/Casting.h"

using namespace cast;

StandardQuantumGate::StandardQuantumGate(
    GateMatrixPtr gateMatrix,
    NoiseChannelPtr noiseChannel,
    const TargetQubitsType& qubits)
  : QuantumGate(QG_Standard) {
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

ScalarGateMatrixPtr StandardQuantumGate::getScalarGM() const {
  if (_gateMatrix == nullptr)
    return nullptr;
  if (llvm::isa<ScalarGateMatrix>(_gateMatrix.get()))
    return std::static_pointer_cast<ScalarGateMatrix>(_gateMatrix);
  return nullptr;
}

std::ostream& StandardQuantumGate::displayInfo(std::ostream& os,
                                               int verbose) const {
  os << BOLDCYAN("=== Info of StandardQuantumGate @ " << this << " === ")
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

/**** op count *****/
namespace {
  double opCount_scalar(const ScalarGateMatrix& matrix, double zeroTol) {
    double count = 0.0;
    size_t len = matrix.matrix().size();
    const auto* data = matrix.matrix().data();
    for (size_t i = 0; i < len; ++i) {
      if (std::abs(data[i]) > zeroTol)
        count += 1.0;
    }
    return count * std::pow<double>(2.0, 1 - matrix.nQubits());
  }
} // anonymous namespace

double StandardQuantumGate::opCount(double zeroTol) const {
  if (_gateMatrix == nullptr)
    return 0.0;

  if (auto* sMat = llvm::dyn_cast<ScalarGateMatrix>(_gateMatrix.get())) {
    return opCount_scalar(*sMat, zeroTol);
  }

  assert(false && "Not Implemented yet");
  return 0.0;
}