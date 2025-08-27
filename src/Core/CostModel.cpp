#include "cast/Core/CostModel.h"

using namespace cast;

double SizeOnlyCostModel::computeGiBTime(const QuantumGate* gate) const {
  assert(gate != nullptr);
  if (gate->nQubits() > maxSize)
    return 1.0;
  if (maxOp > 0 && gate->opCount(zeroTol) > maxOp)
    return 1.0;
  // effectively 0.0
  return 1e-10;
}
