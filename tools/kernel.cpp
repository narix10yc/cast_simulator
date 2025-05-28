#include "cast/CPU/StatevectorCPU.h"
#include "timeit/timeit.h"

#include "cast/Parser.h"
#include "cast/Legacy/CircuitGraph.h"
#include "cast/Fusion.h"
#include "openqasm/parser.h"

using namespace cast;
#define SIMD_S 3

int main(int argc, const char** argv) {
  assert(argc > 2);

  openqasm::Parser qasmParser(argv[1], 0);
  auto qasmRoot = qasmParser.parse();
  int nThreads = std::stoi(argv[2]);

  // This is temporary work-around as CircuitGraph does not allow copy yet
  legacy::CircuitGraph graphNoFuse, graphNaiveFuse, graphAdaptiveFuse;
  qasmRoot->toCircuitGraph(graphNoFuse);
  qasmRoot->toCircuitGraph(graphNaiveFuse);
  qasmRoot->toCircuitGraph(graphAdaptiveFuse);

  FusionConfig fusionConfig = FusionConfig::Aggressive;
  fusionConfig.precision = 64;
  fusionConfig.nThreads = nThreads;
  // NaiveCostModel costModel(3, 0, 1e-8);

  NaiveCostModel naiveCostModel(5, -1, 1e-8);
  applyGateFusion(fusionConfig, &naiveCostModel, graphNaiveFuse);

  auto cache = PerformanceCache::LoadFromCSV("t382.csv") ;
  StandardCostModel standardCostModel(&cache);
  applyGateFusion(fusionConfig, &standardCostModel, graphAdaptiveFuse);

  CPUKernelManager kernelMgr;
  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = SIMD_S;

  kernelGenConfig.displayInfo(std::cerr) << "\n";

  utils::timedExecute([&]() {
    kernelMgr.genCPUGatesFromCircuitGraph(
      kernelGenConfig, graphNoFuse, "graphNoFuse");
  }, "Generate No-fuse Kernels");
  utils::timedExecute([&]() {
    kernelMgr.genCPUGatesFromCircuitGraph(
      kernelGenConfig, graphNaiveFuse, "graphNaiveFuse");
  }, "Generate Naive-fused Kernels");
  utils::timedExecute([&]() {
    kernelMgr.genCPUGatesFromCircuitGraph(
      kernelGenConfig, graphAdaptiveFuse, "graphAdaptiveFuse");
  }, "Generate Adaptive-fused Kernels");

  std::vector<CPUKernelInfo*> kernelsNoFuse, kernelsNaiveFuse, kernelAdaptiveFuse;
  utils::timedExecute([&]() {
    kernelMgr.initJIT(
      nThreads,
      llvm::OptimizationLevel::O1,
      /* useLazyJIT */ false,
      /* verbose */ 1);
    kernelsNoFuse = kernelMgr.collectCPUKernelsFromCircuitGraph("graphNoFuse");
    kernelsNaiveFuse = kernelMgr.collectCPUKernelsFromCircuitGraph("graphNaiveFuse");
    kernelAdaptiveFuse = kernelMgr.collectCPUKernelsFromCircuitGraph("graphAdaptiveFuse");
  }, "JIT compile kernels");

  utils::StatevectorCPU<double> sv(graphNoFuse.nQubits, kernelGenConfig.simd_s);
  // sv.randomize();

  timeit::Timer timer(/* replication */ 1);
  timeit::TimingResult tr;

  tr = timer.timeit([&]() {
    for (auto* kernel : kernelsNoFuse)
      kernelMgr.applyCPUKernelMultithread(sv.data(), sv.nQubits(), *kernel, nThreads);
  });
  tr.display(3, std::cerr << "No-fuse Circuit:\n");

  tr = timer.timeit([&]() {
    for (auto* kernel : kernelsNaiveFuse)
      kernelMgr.applyCPUKernelMultithread(sv.data(), sv.nQubits(), *kernel, nThreads);
  });
  tr.display(3, std::cerr << "Naive-fused Circuit:\n");

  tr = timer.timeit([&]() {
    for (auto* kernel : kernelAdaptiveFuse)
      kernelMgr.applyCPUKernelMultithread(sv.data(), sv.nQubits(), *kernel, nThreads);
  });
  tr.display(3, std::cerr << "Adaptive-fused Circuit:\n");

  return 0;
}