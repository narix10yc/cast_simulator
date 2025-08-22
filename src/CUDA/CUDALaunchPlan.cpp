#include "cast/CUDA/CUDALaunchPlan.h"
#include <algorithm>

namespace cast {

bool streakWorthPermuting(std::span<const GateDesc> gates, size_t i,
                          unsigned lookaheadS,
                          const CostModel& cm,
                          unsigned nQubits, size_t sizeofScalar,
                          double epsilon) {
  const auto& g0 = gates[i];
  if (g0.k() <= 3) return false;
  unsigned S = 1;
  for (size_t j = i+1; j < gates.size() && S < lookaheadS; ++j) {
    if (gates[j].targets == g0.targets) ++S; else break;
  }
  if (S < 2) return false;
  double save_ms = 0.0;
  for (unsigned s = 0; s < S; ++s) {
    unsigned k = g0.k();
    save_ms += std::max(0.0, cm.tGeneric_ms[k] - cm.tLSB_ms[k]);
  }
  double tperm = estimatePermuteMs(nQubits, sizeofScalar, cm.BW_GBps);
  return save_ms >= (1.0 + epsilon) * tperm;
}

} // namespace cast
