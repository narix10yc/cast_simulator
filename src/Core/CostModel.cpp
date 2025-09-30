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

void SizeOnlyCostModel::displayInfo(utils::InfoLogger logger) const {
  logger.put("SizeOnlyCostModel")
      .put("maxSize", maxSize)
      .put("maxOp", maxOp)
      .put("zeroTol", zeroTol);
}

void cast::impl::computeGateWeights(const std::array<float, 5>& tarr,
                                    impl::CostModelWeightType& weights) {

  // The following two parameters decides the scaling of weights
  // when mem speed halves according to a Lorentzian bump.
  constexpr float T = 15.f;
  constexpr float eps = 0.1f;

  // Exponentially decays the weights for >=5 qubit gates
  constexpr float decayLargeGates = 0.45f;

  // weights[k] is the weight of k-qubit gates
  // initialize the weight of 1-qubit gates to 100.0f
  weights[0] = 100.0f;
  for (int k = 2; k <= 5; ++k) {
    // Lorentzian bump
    auto x = tarr[k - 1] / tarr[k - 2] - 1;
    auto y = 1 - (T - 1) * eps + (T - 1) * eps * (1 + eps) / (x * x + eps);
    weights[k - 1] = weights[k - 2] * y;
  }

  // Exponentially decay weights for >=5 qubit gates
  for (int k = 5; k <= weights.size(); ++k) {
    weights[k - 1] = weights[k - 2] * decayLargeGates;
  }
}