#include "cast/Core/QuantumGate.h"
#include "utils/PrintSpan.h"
#include "utils/iocolor.h"

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
  qubits_ = qubits;
}

void SuperopQuantumGate::displayInfo(utils::InfoLogger logger) const {
  logger.put("Superop Quantum Gate")
      .put("Target Qubits", std::span(qubits_))
      .put("Superop Matrix", _superopMatrix.get())
      .put("opCount", opCount(1e-8));
}