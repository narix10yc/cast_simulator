#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "openqasm/parser.h"
#include "openqasm/ast.h"
#include "cast/CircuitGraph.h"
#include "cast/Fusion.h"
#include "simulation/KernelManager.h"
#include "simulation/StatevectorCUDA.h"
#include "timeit/timeit.h"
#include "llvm/Passes/OptimizationLevel.h"

using namespace cast;

enum class FusionStrategy {
  NoFuse,
  Naive,
  Adaptive,
  Cuda
};

static const char* toString(FusionStrategy fs) {
  switch (fs) {
    case FusionStrategy::NoFuse:   return "NoFuse";
    case FusionStrategy::Naive:    return "Naive";
    case FusionStrategy::Adaptive: return "Adaptive";
    case FusionStrategy::Cuda:     return "Cuda";
    default:                       return "Unknown";
  }
}

static void printTimingStats(const std::string& label, const timeit::TimingResult& tr) {
  if (tr.tArr.empty()) {
    std::cout << label << ": No data\n";
    return;
  }
  double sum = 0.0;
  for (auto& t : tr.tArr) sum += t;
  double mean = sum / tr.tArr.size();
  double sumsq = 0.0;
  for (auto& t : tr.tArr) {
    double diff = t - mean;
    sumsq += diff * diff;
  }
  double var = (tr.tArr.size() > 1) ? (sumsq / tr.tArr.size()) : 0.0;
  double stdev = std::sqrt(var);
  auto computeMedian = [&](const std::vector<double>& arr) {
    if (arr.empty()) return 0.0;
    std::vector<double> sorted(arr);
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    if (n % 2 == 1) return sorted[n / 2];
    return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
  };
  double med = computeMedian(tr.tArr);
  double meanMs   = mean   * 1e3;
  double stdevMs  = stdev  * 1e3;
  double medianMs = med    * 1e3;
  std::cout << label << ": mean=" << meanMs << " ms ± " << stdevMs
            << " ms (median=" << medianMs << " ms, n=" << tr.tArr.size() << ")\n";
}

int main() {
  using namespace timeit;

  std::string qasmFile         = "../examples/qft/qft-16-cp.qasm";
  std::string adaptiveModelPath = "cost_model_simd1.csv";
  std::string cudaModelPath     = "cuda128test.csv";
  int naiveMaxK    = 2;
  int nReps        = 5;
  int blockSize    = 128;
  int nThreads     = 1;
  auto optLevel    = llvm::OptimizationLevel::O1;
  FusionStrategy strategy = FusionStrategy::Cuda;

  Timer parseTimer(nReps);
  TimingResult parseTR = parseTimer.timeit([&]() {
    openqasm::Parser parser(qasmFile, 0);
    auto r = parser.parse();
  });
  openqasm::Parser parser(qasmFile, 0);
  auto root = parser.parse();
  if (!root) {
    std::cerr << "Error: Could not parse.\n";
    return 1;
  }

  Timer buildTimer(nReps);
  TimingResult buildTR = buildTimer.timeit([&]() {
    auto tmp = std::make_unique<cast::CircuitGraph>();
    root->toCircuitGraph(*tmp);
  });
  auto graphPtr = std::make_unique<cast::CircuitGraph>();
  root->toCircuitGraph(*graphPtr);

  Timer fusionTimer(nReps);
  TimingResult fusionTR = fusionTimer.timeit([&]() {
    auto tmp = std::make_unique<cast::CircuitGraph>();
    root->toCircuitGraph(*tmp);
    FusionConfig fusionConfig = FusionConfig::Aggressive;
    fusionConfig.nThreads  = -1;
    fusionConfig.precision = 64;
    switch (strategy) {
      case FusionStrategy::NoFuse: break;
      case FusionStrategy::Naive: {
        NaiveCostModel naiveModel(naiveMaxK, -1, 1e-8);
        applyGateFusion(fusionConfig, &naiveModel, *tmp);
      } break;
      case FusionStrategy::Adaptive: {
        if (!adaptiveModelPath.empty()) {
          auto cache = PerformanceCache::LoadFromCSV(adaptiveModelPath);
          StandardCostModel scModel(&cache);
          applyGateFusion(fusionConfig, &scModel, *tmp);
        }
      } break;
      case FusionStrategy::Cuda: {
        if (!cudaModelPath.empty()) {
          auto cudaCache = CUDAPerformanceCache::LoadFromCSV(cudaModelPath);
          CUDACostModel cudaModel(&cudaCache);
          applyGateFusion(fusionConfig, &cudaModel, *tmp);
        }
      } break;
    }
  });
  {
    FusionConfig fusionConfig = FusionConfig::Aggressive;
    fusionConfig.nThreads  = -1;
    fusionConfig.precision = 64;
    switch (strategy) {
      case FusionStrategy::NoFuse: break;
      case FusionStrategy::Naive: {
        NaiveCostModel naiveModel(naiveMaxK, -1, 1e-8);
        applyGateFusion(fusionConfig, &naiveModel, *graphPtr);
      } break;
      case FusionStrategy::Adaptive: {
        if (!adaptiveModelPath.empty()) {
          auto cache = PerformanceCache::LoadFromCSV(adaptiveModelPath);
          StandardCostModel scModel(&cache);
          applyGateFusion(fusionConfig, &scModel, *graphPtr);
        }
      } break;
      case FusionStrategy::Cuda: {
        if (!cudaModelPath.empty()) {
          auto cudaCache = CUDAPerformanceCache::LoadFromCSV(cudaModelPath);
          CUDACostModel cudaModel(&cudaCache);
          applyGateFusion(fusionConfig, &cudaModel, *graphPtr);
        }
      } break;
    }
  }

  CUDAKernelGenConfig genConfig;
  genConfig.blockSize      = blockSize;
  genConfig.precision      = 64;
  genConfig.matrixLoadMode = CUDAKernelGenConfig::UseMatImmValues;

  Timer kernelGenTimer(nReps);
  TimingResult kernelGenTR = kernelGenTimer.timeit([&]() {
    CUDAKernelManager localKM;
    localKM.genCUDAGatesFromCircuitGraph(genConfig, *graphPtr, "testCircuit");
  });

  Timer ptxTimer(nReps);
  TimingResult ptxTR = ptxTimer.timeit([&]() {
    CUDAKernelManager localKM;
    localKM.genCUDAGatesFromCircuitGraph(genConfig, *graphPtr, "testCircuit");
    localKM.emitPTX(nThreads, optLevel, 0);
  });

  Timer jitTimer(nReps);
  TimingResult jitTR = jitTimer.timeit([&]() {
    CUDAKernelManager localKM;
    localKM.genCUDAGatesFromCircuitGraph(genConfig, *graphPtr, "testCircuit");
    localKM.emitPTX(nThreads, optLevel, 0);
    localKM.initCUJIT(nThreads, 0);
  });

  CUDAKernelManager kernelMgr;
  kernelMgr.genCUDAGatesFromCircuitGraph(genConfig, *graphPtr, "testCircuit");
  kernelMgr.emitPTX(nThreads, optLevel, 0);
  kernelMgr.initCUJIT(nThreads, 0);

  auto kernels = kernelMgr.collectCUDAKernelsFromCircuitGraph("testCircuit");

  Timer execTimer(nReps);
  TimingResult execTR = execTimer.timeit([&]() {
    utils::StatevectorCUDA<double> sv(graphPtr->nQubits);
    sv.initialize();
    for (auto* k : kernels) {
      kernelMgr.launchCUDAKernel(sv.dData(), sv.nQubits(), *k, blockSize);
    }
    cudaDeviceSynchronize();
  });

  std::cout << "======== Timing Results ========\n";
  printTimingStats("1) Parse Time", parseTR);
  printTimingStats("2) Build Time", buildTR);
  printTimingStats("3) Fusion Time", fusionTR);
  printTimingStats("4) Kernel Gen Time", kernelGenTR);
  printTimingStats("5) PTX Time", ptxTR);
  printTimingStats("6) JIT Time", jitTR);
  printTimingStats("7) Execution Time", execTR);
  std::cout << "================================\n";
  return 0;
}