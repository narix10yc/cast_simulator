#include "simulation/StatevectorCPU.h"
#include "timeit/timeit.h"

#include "cast/Parser.h"
#include "cast/LegacyCircuitGraph.h"
#include "cast/Fusion.h"
#include "openqasm/parser.h"

#include <llvm/Support/CommandLine.h>

namespace cl = llvm::cl;

using namespace cast;

cl::opt<std::string>
ArgInputFilename("i",
  cl::desc("Input file name"), cl::Positional, cl::Required);

cl::opt<std::string>
ArgModelPath("model",
  cl::desc("Path to performance model"), cl::init(""));

cl::opt<int>
ArgNThreads("T", cl::desc("Number of threads"), cl::Prefix, cl::Required);

cl::opt<int>
ArgSimd_s("simd-s", cl::desc("simd s"), cl::init(1));

cl::opt<bool>
ArgRunNoFuse("run-no-fuse", cl::desc("Run no-fuse circuit"), cl::init(false));

cl::opt<bool>
ArgRunNaiveFuse("run-naive-fuse",
  cl::desc("Run naive-fuse circuit"), cl::init(false));

cl::opt<bool>
ArgRunAdaptiveFuse("run-adaptive-fuse",
  cl::desc("Run adaptive-fuse circuit"), cl::init(true));

cl::opt<int>
ArgNaiveMaxK("naive-max-k",
  cl::desc("The max size of gates in naive fusion"), cl::init(3));

cl::opt<int>
ArgReplication("replication",
  cl::desc("Number of replications"), cl::init(1));

int main(int argc, const char** argv) {
  cl::ParseCommandLineOptions(argc, argv);

  openqasm::Parser qasmParser(ArgInputFilename, 0);
  auto qasmRoot = qasmParser.parse();

  // This is temporary work-around as LegacyCircuitGraph does not allow copy yet
  LegacyCircuitGraph graphNoFuse, graphNaiveFuse, graphAdaptiveFuse;
  qasmRoot->toLegacyCircuitGraph(graphNoFuse);
  qasmRoot->toLegacyCircuitGraph(graphNaiveFuse);
  qasmRoot->toLegacyCircuitGraph(graphAdaptiveFuse);

  FusionConfig fusionConfig = FusionConfig::Aggressive;
  fusionConfig.precision = 64;
  fusionConfig.nThreads = ArgNThreads;
  if (ArgRunNaiveFuse) {
    NaiveCostModel naiveCostModel(ArgNaiveMaxK, -1, 1e-8);
    applyGateFusion(fusionConfig, &naiveCostModel, graphNaiveFuse);
  }

  if (ArgModelPath != "" && ArgRunAdaptiveFuse) {
    auto cache = PerformanceCache::LoadFromCSV(ArgModelPath);
    StandardCostModel standardCostModel(&cache);
    standardCostModel.display(std::cerr); 
    applyGateFusion(fusionConfig, &standardCostModel, graphAdaptiveFuse);
  }

  CPUKernelManager kernelMgr;
  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = ArgSimd_s;
  kernelGenConfig.displayInfo(std::cerr) << "\n";

  // Generate kernels
  if (ArgRunNoFuse) {
    utils::timedExecute([&]() {
      kernelMgr.genCPUGatesFromLegacyCircuitGraph(
        kernelGenConfig, graphNoFuse, "graphNoFuse");
    }, "Generate No-fuse Kernels");
  }
  if (ArgRunNaiveFuse) {
    utils::timedExecute([&]() {
      kernelMgr.genCPUGatesFromLegacyCircuitGraph(
        kernelGenConfig, graphNaiveFuse, "graphNaiveFuse");
    }, "Generate Naive-fused Kernels");
  }
  if (ArgModelPath != "" && ArgRunAdaptiveFuse) {
    utils::timedExecute([&]() {
      kernelMgr.genCPUGatesFromLegacyCircuitGraph(
        kernelGenConfig, graphAdaptiveFuse, "graphAdaptiveFuse");
    }, "Generate Adaptive-fused Kernels");
  }

  // JIT compile kernels
  std::vector<CPUKernelInfo*> kernelsNoFuse;
  std::vector<CPUKernelInfo*> kernelsNaiveFuse;
  std::vector<CPUKernelInfo*>kernelAdaptiveFuse;

  utils::timedExecute([&]() {
    kernelMgr.initJIT(
      ArgNThreads,
      llvm::OptimizationLevel::O1,
      /* useLazyJIT */ false,
      /* verbose */ 1);
    double opCountTotal = 0.0;
    if (ArgRunNoFuse) {
      opCountTotal = 0.0;
      kernelsNoFuse = 
        kernelMgr.collectCPUKernelsFromLegacyCircuitGraph("graphNoFuse");
      for (const auto* kernel : kernelsNoFuse)
        opCountTotal += kernel->gate->opCount(1e-8);
      std::cerr << "No-fuse: nGates = " << kernelsNoFuse.size()
                << "; opCount = " << opCountTotal << "\n";
    }
    if (ArgRunNaiveFuse) {
      opCountTotal = 0.0;
      kernelsNaiveFuse = 
        kernelMgr.collectCPUKernelsFromLegacyCircuitGraph("graphNaiveFuse");
      for (const auto* kernel : kernelsNaiveFuse)
        opCountTotal += kernel->gate->opCount(1e-8);
      std::cerr << "Naive-fuse: nGates = " << kernelsNaiveFuse.size()
                << "; opCount = " << opCountTotal << "\n";
    }
    if (ArgRunAdaptiveFuse) {
      if (ArgModelPath == "") {
        std::cerr << RED("ERROR: Adaptive fusion requires -model <file>. Skipping.\n");
        return;
      }
      opCountTotal = 0.0;
      kernelAdaptiveFuse = 
        kernelMgr.collectCPUKernelsFromLegacyCircuitGraph("graphAdaptiveFuse");
      for (const auto* kernel : kernelAdaptiveFuse)
        opCountTotal += kernel->gate->opCount(1e-8);
      std::cerr << "Adaptive-fuse: nGates = " << kernelAdaptiveFuse.size()
                << "; opCount = " << opCountTotal << "\n";
    }
    }, "JIT compile kernels");

  // Run kernels
  utils::StatevectorCPU<double> sv(graphNoFuse.nQubits, kernelGenConfig.simd_s);
  // sv.randomize();
  timeit::Timer timer(ArgReplication);
  timeit::TimingResult tr;

  if (ArgRunNoFuse) {
    tr = timer.timeit([&]() {
      for (auto* kernel : kernelsNoFuse) {
        kernelMgr.applyCPUKernelMultithread(
          sv.data(), sv.nQubits(), *kernel, ArgNThreads);
      }
    });
    tr.display(3, std::cerr << "No-fuse Circuit:\n");
  }

  if (ArgRunNaiveFuse) {
    tr = timer.timeit([&]() {
      for (auto* kernel : kernelsNaiveFuse) {
        kernelMgr.applyCPUKernelMultithread(
          sv.data(), sv.nQubits(), *kernel, ArgNThreads);
      }
    });
    tr.display(3, std::cerr << "Naive-fused Circuit:\n");
  }
  
  if (!kernelAdaptiveFuse.empty()) {
    tr = timer.timeit([&]() {
      for (auto* kernel : kernelAdaptiveFuse) {
        kernelMgr.applyCPUKernelMultithread(
          sv.data(), sv.nQubits(), *kernel, ArgNThreads);
      }
    });
    tr.display(3, std::cerr << "Adaptive-fused Circuit:\n");
  }
  return 0;
}