#include "cast/CPU/CPUCostModel.h"
#include "cast/CPU/CPUKernelManager.h"

using namespace cast;

int main() {
  CPUPerformanceCache cache;
  CPUPerformanceCache::WeightType weights;
  CPUKernelGenConfig config(CPUSimdWidth::W128, Precision::F64);
  cache.runPreliminaryExperiments(CPUKernelGenConfig(), 28, 10, weights, 3);
}