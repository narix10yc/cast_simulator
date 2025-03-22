#include "simulation/StatevectorCUDA.h"
#include "timeit/timeit.h"

#include "cast/Parser.h"
#include "cast/CircuitGraph.h"
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

cl::opt<bool>
ArgRunNoFuse("run-no-fuse", cl::desc("Run no-fuse circuit"), cl::init(false));

cl::opt<bool>
ArgRunNaiveFuse("run-naive-fuse",
  cl::desc("Run naive-fuse circuit"), cl::init(true));

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

  // This is temporary work-around as CircuitGraph does not allow copy yet
  CircuitGraph graphNoFuse, graphNaiveFuse, graphAdaptiveFuse;
  qasmRoot->toCircuitGraph(graphNoFuse);
  qasmRoot->toCircuitGraph(graphNaiveFuse);
  qasmRoot->toCircuitGraph(graphAdaptiveFuse);

  FusionConfig fusionConfigAggresive = FusionConfig::Aggressive;
  fusionConfigAggresive.precision = 64;
  fusionConfigAggresive.nThreads = -1; // setting to -1 means GPU
  if (ArgRunNaiveFuse) {
    NaiveCostModel naiveCostModel(ArgNaiveMaxK, -1, 1e-8);
    applyGateFusion(fusionConfigAggresive, &naiveCostModel, graphNaiveFuse);
  }

  if (ArgRunAdaptiveFuse && ArgModelPath != "") {
    auto cache = PerformanceCache::LoadFromCSV(ArgModelPath);
    StandardCostModel standardCostModel(&cache);
    standardCostModel.display(std::cerr);
    applyGateFusion(fusionConfigAggresive, &standardCostModel, graphAdaptiveFuse);
  }

  CUDAKernelManager kernelMgr;
  CUDAKernelGenConfig kernelGenConfig;

  kernelGenConfig.displayInfo(std::cerr) << "\n";

  // Generate kernels
  if (ArgRunNoFuse) {
    utils::timedExecute([&]() {
      kernelMgr.genCUDAGatesFromCircuitGraph(
        kernelGenConfig, graphNoFuse, "graphNoFuse");
    }, "Generate No-fuse Kernels");
  }
  if (ArgRunNaiveFuse) {
    utils::timedExecute([&]() {
      kernelMgr.genCUDAGatesFromCircuitGraph(
        kernelGenConfig, graphNaiveFuse, "graphNaiveFuse");
    }, "Generate Naive-fused Kernels");
  }
  if (ArgRunAdaptiveFuse && ArgModelPath != "") {
    utils::timedExecute([&]() {
      kernelMgr.genCUDAGatesFromCircuitGraph(
        kernelGenConfig, graphAdaptiveFuse, "graphAdaptiveFuse");
    }, "Generate Adaptive-fused Kernels");
  }

  // JIT compile kernels
  std::vector<CUDAKernelInfo*> kernelsNoFuse, kernelsNaiveFuse, kernelAdaptiveFuse;
  utils::timedExecute([&]() {
    kernelMgr.emitPTX(ArgNThreads, llvm::OptimizationLevel::O1, 1);
    kernelMgr.initCUJIT(ArgNThreads, 1);
    double opCountTotal = 0.0;
    if (ArgRunNoFuse) {
      opCountTotal = 0.0;
      kernelsNoFuse = 
        kernelMgr.collectCUDAKernelsFromCircuitGraph("graphNoFuse");
      for (const auto* kernel : kernelsNoFuse)
        opCountTotal += kernel->gate->opCount(1e-8);
      std::cerr << "No-fuse: nGates = " << kernelsNoFuse.size()
                << "; opCount = " << opCountTotal << "\n";
    }
    if (ArgRunNaiveFuse) {
      opCountTotal = 0.0;
      kernelsNaiveFuse = 
        kernelMgr.collectCUDAKernelsFromCircuitGraph("graphNaiveFuse");
      for (const auto* kernel : kernelsNaiveFuse)
        opCountTotal += kernel->gate->opCount(1e-8);
      std::cerr << "Naive-fuse: nGates = " << kernelsNaiveFuse.size()
                << "; opCount = " << opCountTotal << "\n";
    }
    if (ArgRunAdaptiveFuse && ArgModelPath != "") {
      opCountTotal = 0.0;
      kernelAdaptiveFuse = 
        kernelMgr.collectCUDAKernelsFromCircuitGraph("graphAdaptiveFuse");
      for (const auto* kernel : kernelAdaptiveFuse)
        opCountTotal += kernel->gate->opCount(1e-8);
      std::cerr << "Adaptive-fuse: nGates = " << kernelAdaptiveFuse.size()
                << "; opCount = " << opCountTotal << "\n";
    }
    }, "JIT compile kernels");

  // Run kernels
  utils::StatevectorCUDA<double> sv(graphNoFuse.nQubits);
  // sv.randomize();
  sv.initialize();
  timeit::Timer timer(ArgReplication);
  timeit::TimingResult tr;

  if (ArgRunNoFuse) {
    tr = timer.timeit([&]() {
      for (auto* kernel : kernelsNoFuse)
        kernelMgr.launchCUDAKernel(sv.dData(), sv.nQubits(), *kernel);
      cudaDeviceSynchronize();
    });
    tr.display(3, std::cerr << "No-fuse Circuit:\n");
  }

  if (ArgRunNaiveFuse) {
    tr = timer.timeit([&]() {
      for (auto* kernel : kernelsNaiveFuse)
        kernelMgr.launchCUDAKernel(sv.dData(), sv.nQubits(), *kernel);
      cudaDeviceSynchronize();
    });
    tr.display(3, std::cerr << "Naive-fused Circuit:\n");
  }
  if (ArgRunAdaptiveFuse && ArgModelPath != "") {
    tr = timer.timeit([&]() {
      for (auto* kernel : kernelAdaptiveFuse)
        kernelMgr.launchCUDAKernel(sv.dData(), sv.nQubits(), *kernel);
      cudaDeviceSynchronize();
    });
    tr.display(3, std::cerr << "Adaptive-fused Circuit:\n");
  }

  return 0;
}