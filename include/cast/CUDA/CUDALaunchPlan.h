#ifndef CAST_CUDA_CUDALAUNCHPLAN_H
#define CAST_CUDA_CUDALAUNCHPLAN_H

#include <vector>
#include <span>
#include <array>
#include "cast/CUDA/CUDALayout.h"

namespace cast {

struct GateDesc {
  std::vector<int> targets;   // logical ids
  unsigned k() const { return (unsigned)targets.size(); }
  // plus ptrs to matrix if needed
};

struct CostModel {
  double BW_GBps = 800.0;              // set at init from a quick probe
  std::array<double, 9> tLSB_ms{};     // index by k (0..8), fill used ks
  std::array<double, 9> tGeneric_ms{};
};

inline double estimatePermuteMs(unsigned n, size_t sizeofScalar, double BW_GBps) {
  const double bytes = 2.0 * (2.0 * sizeofScalar * (double)(1ull<<n));
  return (bytes / (BW_GBps * 1e9)) * 1e3;
}

bool streakWorthPermuting(std::span<const GateDesc> gates, size_t i,
                          unsigned lookaheadS,
                          const CostModel& cm,
                          unsigned nQubits, size_t sizeofScalar,
                          double epsilon = 0.05);

enum class PlanKind { ExecLSB, ExecGeneric, PermuteThenExecLSB };

struct CompiledKernelRef {
  // string name that the launcher uses
  std::string llvmFuncName;
};

struct Plan {
  PlanKind kind;
  std::vector<AxisSwap> swaps;
  CompiledKernelRef kernel;
};

} // namespace cast

#endif // CAST_CUDA_CUDALAUNCHPLAN_H