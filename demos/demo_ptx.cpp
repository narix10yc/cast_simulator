#include "cast/CUDA/StatevectorCUDA.h"
#include "timeit/timeit.h"

#include "cast/Parser.h"
#include "cast/Legacy/CircuitGraph.h"
#include "cast/Legacy/CircuitGraph.h"
#include "cast/Fusion.h"
#include "openqasm/parser.h"

#include "llvm/Support/CommandLine.h>

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
ArgBlockSize("blocksize",
  cl::desc("Block size used in CUDA kernels (min 32, max 512, in powers of 2)"),
  cl::init(64));

cl::opt<int>
ArgReplication("replication",
  cl::desc("Number of replications"), cl::init(5));

cl::opt<std::string>
ArgCUDAModelPath("cuda-model",
  cl::desc("Path to CUDA performance model"), cl::init(""));

cl::opt<bool>
ArgRunCudaFuse("run-cuda-fuse",
  cl::desc("Run CUDA-optimized fusion"), cl::init(false));

int main(int argc, const char** argv) {
  cl::ParseCommandLineOptions(argc, argv);

  openqasm::Parser qasmParser(ArgInputFilename, 0);
  auto qasmRoot = qasmParser.parse();

  // This is temporary work-around as legacy::CircuitGraph does not allow copy yet
  legacy::CircuitGraph graphNoFuse, graphNaiveFuse, graphAdaptiveFuse, graphCudaFuse;
  qasmRoot->tolegacy::CircuitGraph(graphNoFuse);
  qasmRoot->tolegacy::CircuitGraph(graphNaiveFuse);
  qasmRoot->tolegacy::CircuitGraph(graphAdaptiveFuse);
  qasmRoot->tolegacy::CircuitGraph(graphCudaFuse);

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

  if (ArgRunCudaFuse && ArgCUDAModelPath != "") {
    auto cudaCache = CUDAPerformanceCache::LoadFromCSV(ArgCUDAModelPath);
    CUDACostModel cudaCostModel(&cudaCache);
    cudaCostModel.setBlockSize(ArgBlockSize);
    // standardCostModel.display(std::cerr);
    applyGateFusion(fusionConfigAggresive, &cudaCostModel, graphCudaFuse);
  }

  CUDAKernelManager kernelMgr;
  CUDAKernelGenConfig kernelGenConfig;
  kernelGenConfig.blockSize = ArgBlockSize;

  kernelGenConfig.displayInfo(std::cerr) << "\n";

  // Generate kernels
  if (ArgRunNoFuse) {
    utils::timedExecute([&]() {
      kernelMgr.genCUDAGatesFromlegacy::CircuitGraph(
        kernelGenConfig, graphNoFuse, "graphNoFuse");
    }, "Generate No-fuse Kernels");
  }
  if (ArgRunNaiveFuse) {
    utils::timedExecute([&]() {
      kernelMgr.genCUDAGatesFromlegacy::CircuitGraph(
        kernelGenConfig, graphNaiveFuse, "graphNaiveFuse");
    }, "Generate Naive-fused Kernels");
  }
  if (ArgRunAdaptiveFuse && ArgModelPath != "") {
    utils::timedExecute([&]() {
      kernelMgr.genCUDAGatesFromlegacy::CircuitGraph(
        kernelGenConfig, graphAdaptiveFuse, "graphAdaptiveFuse");
    }, "Generate Adaptive-fused Kernels");
  }
  if (ArgRunCudaFuse && ArgCUDAModelPath != "") {
    utils::timedExecute([&]() {
      kernelMgr.genCUDAGatesFromlegacy::CircuitGraph(
        kernelGenConfig, graphCudaFuse, "graphCudaFuse");
    }, "Generate CUDA-optimized Kernels");
  }

  // JIT compile kernels
  std::vector<CUDAKernelInfo*> kernelsNoFuse, kernelsNaiveFuse, kernelAdaptiveFuse, kernelCudaFuse;
  utils::timedExecute([&]() {
    kernelMgr.emitPTX(ArgNThreads, llvm::OptimizationLevel::O1, 1);
    kernelMgr.initCUJIT(ArgNThreads, 1);
    auto printStats = [&](const std::string& name, 
                         const std::vector<CUDAKernelInfo*>& kernels) {
      double opCountTotal = 0.0;
      for (const auto* kernel : kernels)
        opCountTotal += kernel->gate->opCount(1e-8);
      std::cerr << name << ": nGates = " << kernels.size()
                << "; opCount = " << opCountTotal << "\n";
    };
    double opCountTotal = 0.0;
    if (ArgRunNoFuse) {
      opCountTotal = 0.0;
      kernelsNoFuse = 
        kernelMgr.collectCUDAKernelsFromlegacy::CircuitGraph("graphNoFuse");
      for (const auto* kernel : kernelsNoFuse)
        opCountTotal += kernel->gate->opCount(1e-8);
      std::cerr << "No-fuse: nGates = " << kernelsNoFuse.size()
                << "; opCount = " << opCountTotal << "\n";
    }
    if (ArgRunNaiveFuse) {
      opCountTotal = 0.0;
      kernelsNaiveFuse = 
        kernelMgr.collectCUDAKernelsFromlegacy::CircuitGraph("graphNaiveFuse");
      for (const auto* kernel : kernelsNaiveFuse)
        opCountTotal += kernel->gate->opCount(1e-8);
      std::cerr << "Naive-fuse: nGates = " << kernelsNaiveFuse.size()
                << "; opCount = " << opCountTotal << "\n";
    }
    if (ArgRunAdaptiveFuse && ArgModelPath != "") {
      opCountTotal = 0.0;
      kernelAdaptiveFuse = 
        kernelMgr.collectCUDAKernelsFromlegacy::CircuitGraph("graphAdaptiveFuse");
      for (const auto* kernel : kernelAdaptiveFuse)
        opCountTotal += kernel->gate->opCount(1e-8);
      std::cerr << "Adaptive-fuse: nGates = " << kernelAdaptiveFuse.size()
                << "; opCount = " << opCountTotal << "\n";
    }
    if (ArgRunCudaFuse && ArgCUDAModelPath != "") {
      opCountTotal = 0.0;
      kernelCudaFuse = 
        kernelMgr.collectCUDAKernelsFromlegacy::CircuitGraph("graphCudaFuse");
      for (const auto* kernel : kernelCudaFuse)
        opCountTotal += kernel->gate->opCount(1e-8);
      std::cerr << "Cuda-fuse: nGates = " << kernelCudaFuse.size()
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
        kernelMgr.launchCUDAKernel(
          sv.dData(), sv.nQubits(), *kernel, ArgBlockSize);
      cudaDeviceSynchronize();
    });
    tr.display(3, std::cerr << "No-fuse Circuit:\n");
  }

  if (ArgRunNaiveFuse) {
    tr = timer.timeit([&]() {
      for (auto* kernel : kernelsNaiveFuse)
        kernelMgr.launchCUDAKernel(
          sv.dData(), sv.nQubits(), *kernel, ArgBlockSize);
      cudaDeviceSynchronize();
    });
    tr.display(3, std::cerr << "Naive-fused Circuit:\n");
  }
  if (ArgRunAdaptiveFuse && ArgModelPath != "") {
    tr = timer.timeit([&]() {
      for (auto* kernel : kernelAdaptiveFuse)
        kernelMgr.launchCUDAKernel(
          sv.dData(), sv.nQubits(), *kernel, ArgBlockSize);
      cudaDeviceSynchronize();
    });
    tr.display(3, std::cerr << "Adaptive-fused Circuit:\n");
  }
  if (ArgRunCudaFuse && ArgCUDAModelPath != "") {
    tr = timer.timeit([&]() {
      for (auto* kernel : kernelCudaFuse)
        kernelMgr.launchCUDAKernel(
          sv.dData(), sv.nQubits(), *kernel, ArgBlockSize);
      cudaDeviceSynchronize();
    });
    tr.display(3, std::cerr << "Cuda-fused Circuit:\n");
  }

  return 0;
}