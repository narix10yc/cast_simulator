#include "cast/Core/QuantumGate.h"
#include "utils/iocolor.h"
#include "utils/PrintSpan.h"

using namespace cast;

static bool isAscending(const std::vector<int>& qubits) {
  for (unsigned i = 1, S = qubits.size(); i < S; ++i) {
    if (qubits[i] <= qubits[i - 1])
      return false;
  }
  return true;
}

SuperopQuantumGate::SuperopQuantumGate(ScalarGateMatrixPtr matrix,
                                       const TargetQubitsType& qubits)
  : QuantumGate(QG_Superop) {
  assert(matrix != nullptr && "Initializing with a null matrix");
  assert(matrix->nQubits() == 2 * qubits.size() &&
         "SuperopQuantumGate requires a 2n-qubit matrix");
  assert(isAscending(qubits) && "Qubits must be sorted in ascending order");
  _superopMatrix = std::move(matrix);
  _qubits = qubits;
}

std::ostream& SuperopQuantumGate::displayInfo(std::ostream& os,
                                              int verbose) const {
  os << BOLDCYAN("=== Info of SuperopQuantumGate @ " << this << " === ")
     << "(Verbose " << verbose << ")\n";
  
  os << CYAN("- Target Qubits: ");
  utils::printSpan(os, std::span<const int>(_qubits)) << "\n";

  os << CYAN("- superopMatrix: ") << _superopMatrix.get() << "\n";
  if (verbose > 1) {
    _superopMatrix->matrix().print(os);
  }
  
  return os;
}