#include "cast/QuantumGate.h"
#include "llvm/Support/Casting.h"
#include "utils/iocolor.h"

using namespace cast;

std::ostream& QuantumGate::displayInfo(std::ostream& os, int verbose) const {
  os << BOLDCYAN("=== Info of QuantumGate @ " << this << " === ")
     << "(Verbose " << verbose << ")\n";
  
  os << CYAN("- nQubits: ") << nQubits() << "\n";
  // gate matrix
  os << CYAN("- gateMatrix: ");
  if (gateMatrix == nullptr)
    os << "nullptr";
  else if (auto* sMat = llvm::dyn_cast<ScalarGateMatrix>(gateMatrix.get()))
    os << "ScalarGateMatrix @ " << sMat;
  else if (auto* uMat = llvm::dyn_cast<UnitaryPermGateMatrix>(gateMatrix.get()))
    os << "UnitaryPermGateMatrix @ " << uMat;
  else if (auto* pMat = llvm::dyn_cast<ParametrizedGateMatrix>(gateMatrix.get()))
    os << "ParametrizedGateMatrix @ " << pMat;
  else
    assert(false && "Unknown GateMatrix type");
  os << "\n";

  if (verbose > 1 && gateMatrix != nullptr) {
    if (auto* sMat = llvm::dyn_cast<ScalarGateMatrix>(gateMatrix.get()))
      sMat->matrix().print(os);
  }

  // noise channel
  os << CYAN("- noiseChannel: ");
  if (noiseChannel == nullptr)
    os << "nullptr";
  else
    os << "NoiseChannel @ " << noiseChannel.get();
  os << "\n";
  if (verbose > 1 && noiseChannel != nullptr) {
    noiseChannel->displayInfo(os, verbose - 1);
  }
  os << BOLDCYAN("========== End ==========\n");
  return os;
}