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

cl::opt<bool>
ArgRunDenseKernel("run-dense-kernel",
  cl::desc("Run dense kernel"), cl::init(false));

cl::opt<int>
ArgReplication("replication",
  cl::desc("Number of replications"), cl::init(1));

static double collectKernelsAndGetTotalOpCount(
    CPUKernelManager& kernelMgr,
    const std::string& graphName,
    std::vector<CPUKernelInfo*>& kernels) {
  double opCountTotal = 0.0;
  kernels = kernelMgr.collectCPUKernelsFromCircuitGraph(graphName);
  for (const auto* kernel : kernels)
    opCountTotal += kernel->gate->opCount(1e-8);
  return opCountTotal;
}

static double collectDenseKernelsAndGetTotalOpCount(
    CPUKernelManager& kernelMgr,
    const std::string& graphName,
    std::vector<CPUKernelInfo*>& kernels) {
  double opCountTotal = 0.0;
  kernels = kernelMgr.collectCPUKernelsFromCircuitGraph(graphName);
  for (const auto* kernel : kernels)
    opCountTotal += std::pow(2.0, kernel->gate->nQubits() + 2);
  return opCountTotal;
}

int main(int argc, const char** argv) {
  cl::ParseCommandLineOptions(argc, argv);

  if (ArgRunAdaptiveFuse && ArgModelPath == "") {
    std::cerr << BOLDRED("[Err] ") 
              << "--model=<path> must be specified when using adaptive fusion.\n";
    return 1;
  }

  if (!(ArgRunNoFuse || ArgRunNaiveFuse || ArgRunAdaptiveFuse)) {
    std::cerr << BOLDRED("[Err] ") 
              << "At least one of --run-no-fuse, --run-naive-fuse, "
              << "--run-adaptive-fuse must be specified.\n";
    return 1;
  }

  openqasm::Parser qasmParser(ArgInputFilename, 0);
  auto qasmRoot = qasmParser.parse();

  // This is temporary work-around as CircuitGraph does not allow copy yet
  LegacyCircuitGraph graphNoFuse, graphNaiveFuse, graphAdaptiveFuse;
  LegacyCircuitGraph graphNoFuseDense, graphNaiveFuseDense, graphAdaptiveFuseDense;

  // we always generate the no-fuse graph to get the number of qubits
  qasmRoot->toCircuitGraph(graphNoFuse);
  if (ArgRunNoFuse) {
    if (ArgRunDenseKernel) {
      qasmRoot->toCircuitGraph(graphNoFuseDense);
    }
  }
  if (ArgRunNaiveFuse) {
    qasmRoot->toCircuitGraph(graphNaiveFuse);
    if (ArgRunDenseKernel) {
      qasmRoot->toCircuitGraph(graphNaiveFuseDense);
    }
  }
  if (ArgRunAdaptiveFuse) {
    qasmRoot->toCircuitGraph(graphAdaptiveFuse);
    if (ArgRunDenseKernel) {
      qasmRoot->toCircuitGraph(graphAdaptiveFuseDense);
    }
  }

  // Fusion
  FusionConfig fusionConfig = FusionConfig::Aggressive;
  fusionConfig.precision = 64;
  fusionConfig.nThreads = ArgNThreads;
  if (ArgRunNaiveFuse) {
    NaiveCostModel naiveCostModel(ArgNaiveMaxK, -1, 1e-8);
    applyGateFusion(fusionConfig, &naiveCostModel, graphNaiveFuse);
    if (ArgRunDenseKernel)
      applyGateFusion(fusionConfig, &naiveCostModel, graphNaiveFuseDense);
  }

  if (ArgModelPath != "" && ArgRunAdaptiveFuse) {
    auto cache = PerformanceCache::LoadFromCSV(ArgModelPath);
    StandardCostModel standardCostModel(&cache);
    standardCostModel.display(std::cerr); 
    applyGateFusion(fusionConfig, &standardCostModel, graphAdaptiveFuse);
    if (ArgRunDenseKernel)
      applyGateFusion(fusionConfig, &standardCostModel, graphAdaptiveFuseDense);
  }

  CPUKernelManager kernelMgr;
  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = ArgSimd_s;
  kernelGenConfig.displayInfo(std::cerr) << "\n";
  auto denseKernelGenConfig = kernelGenConfig;
  denseKernelGenConfig.forceDenseKernel = true;
  denseKernelGenConfig.matrixLoadMode = CPUKernelGenConfig::StackLoadMatElems;

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
  if (ArgRunNoFuse && ArgRunDenseKernel) {
    utils::timedExecute([&]() {
      kernelMgr.genCPUGatesFromCircuitGraph(
        denseKernelGenConfig, graphNoFuseDense, "graphNoFuseDense");
    }, "Generate No-fuse Dense Kernels");
  }
  if (ArgRunNaiveFuse && ArgRunDenseKernel) {
    utils::timedExecute([&]() {
      kernelMgr.genCPUGatesFromCircuitGraph(
        denseKernelGenConfig, graphNaiveFuseDense, "graphNaiveFuseDense");
    }, "Generate Naive-fused Dense Kernels");
  }
  if (ArgModelPath != "" && ArgRunAdaptiveFuse && ArgRunDenseKernel) {
    utils::timedExecute([&]() {
      kernelMgr.genCPUGatesFromCircuitGraph(
        denseKernelGenConfig, graphAdaptiveFuseDense, "graphAdaptiveFuseDense");
    }, "Generate Adaptive-fused Dense Kernels");
  }

  // JIT compile kernels
  std::vector<CPUKernelInfo*> kernelsNoFuse;
  std::vector<CPUKernelInfo*> kernelsNaiveFuse;
  std::vector<CPUKernelInfo*> kernelAdaptiveFuse;
  std::vector<CPUKernelInfo*> denseKernelsNoFuse;
  std::vector<CPUKernelInfo*> denseKernelsNaiveFuse;
  std::vector<CPUKernelInfo*> denseKernelAdaptiveFuse;

  utils::timedExecute([&]() {
    kernelMgr.initJIT(
      ArgNThreads,
      llvm::OptimizationLevel::O1,
      /* useLazyJIT */ false,
      /* verbose */ 1
    );
  }, "JIT compile kernels");

  // Display summaries of kernels under different configurations
  std::cerr << BOLDCYAN("Kernel Summary:\n");
  if (ArgRunNoFuse) {
    std::cerr << "- No-fuse kernels:\n";
    double opCountTotal = collectKernelsAndGetTotalOpCount(
      kernelMgr, "graphNoFuse", kernelsNoFuse);
    std::cerr << "  nGates = " << kernelsNoFuse.size()
              << "; opCount = " << opCountTotal << "\n";
    if (ArgRunDenseKernel) {
      opCountTotal = collectDenseKernelsAndGetTotalOpCount(
        kernelMgr, "graphNoFuseDense", denseKernelsNoFuse);
      std::cerr << "  Dense nGates = " << denseKernelsNoFuse.size()
                << "; opCount = " << opCountTotal << "\n";
    }
  }
  if (ArgRunNaiveFuse) {
    std::cerr << "- Naive-fused kernels:\n";
    double opCountTotal = collectKernelsAndGetTotalOpCount(
      kernelMgr, "graphNaiveFuse", kernelsNaiveFuse);
    std::cerr << "  nGates = " << kernelsNaiveFuse.size()
              << "; opCount = " << opCountTotal << "\n";
    if (ArgRunDenseKernel) {
      opCountTotal = collectDenseKernelsAndGetTotalOpCount(
        kernelMgr, "graphNaiveFuseDense", denseKernelsNaiveFuse);
      std::cerr << "  Dense nGates = " << denseKernelsNaiveFuse.size()
                << "; opCount = " << opCountTotal << "\n";
    }
  }
  if (ArgRunAdaptiveFuse) {
    std::cerr << "- Adaptive-fused kernels:\n";
    double opCountTotal = collectKernelsAndGetTotalOpCount(
      kernelMgr, "graphAdaptiveFuse", kernelAdaptiveFuse);
    std::cerr << "  nGates = " << kernelAdaptiveFuse.size()
              << "; opCount = " << opCountTotal << "\n";
    if (ArgRunDenseKernel) {
      opCountTotal = collectDenseKernelsAndGetTotalOpCount(
        kernelMgr, "graphAdaptiveFuseDense", denseKernelAdaptiveFuse);
      std::cerr << "  Dense nGates = " << denseKernelAdaptiveFuse.size()
                << "; opCount = " << opCountTotal << "\n";
    }
  }

  // Run kernels
  utils::StatevectorCPU<double> sv(graphNoFuse.nQubits, kernelGenConfig.simd_s);
  // sv.randomize();
  timeit::Timer timer(ArgReplication);
  timeit::TimingResult tr;

  std::cerr << BOLDCYAN("Running kernels:\n");
  if (ArgRunNoFuse) {
    tr = timer.timeit([&]() {
      for (auto* kernel : kernelsNoFuse) {
        kernelMgr.applyCPUKernelMultithread(
          sv.data(), sv.nQubits(), *kernel, ArgNThreads);
      }
    });
    tr.display(3, std::cerr << "No-fuse Circuit:\n");
    if (ArgRunDenseKernel) {
      tr = timer.timeit([&]() {
        for (auto* kernel : denseKernelsNoFuse) {
          kernelMgr.applyCPUKernelMultithread(
            sv.data(), sv.nQubits(), *kernel, ArgNThreads);
        }
      });
      tr.display(3, std::cerr << "No-fuse Dense Circuit:\n");
    }
  }

  if (ArgRunNaiveFuse) {
    tr = timer.timeit([&]() {
      for (auto* kernel : kernelsNaiveFuse) {
        kernelMgr.applyCPUKernelMultithread(
          sv.data(), sv.nQubits(), *kernel, ArgNThreads);
      }
    });
    tr.display(3, std::cerr << "Naive-fused Circuit:\n");
    if (ArgRunDenseKernel) {
      tr = timer.timeit([&]() {
        for (auto* kernel : denseKernelsNaiveFuse) {
          kernelMgr.applyCPUKernelMultithread(
            sv.data(), sv.nQubits(), *kernel, ArgNThreads);
        }
      });
      tr.display(3, std::cerr << "Naive-fused Dense Circuit:\n");
    }
  }
  
  if (!kernelAdaptiveFuse.empty()) {
    tr = timer.timeit([&]() {
      for (auto* kernel : kernelAdaptiveFuse) {
        kernelMgr.applyCPUKernelMultithread(
          sv.data(), sv.nQubits(), *kernel, ArgNThreads);
      }
    });
    tr.display(3, std::cerr << "Adaptive-fused Circuit:\n");
    if (ArgRunDenseKernel) {
      tr = timer.timeit([&]() {
        for (auto* kernel : denseKernelAdaptiveFuse) {
          kernelMgr.applyCPUKernelMultithread(
            sv.data(), sv.nQubits(), *kernel, ArgNThreads);
        }
      });
      tr.display(3, std::cerr << "Adaptive-fused Dense Circuit:\n");
    }
  }
  return 0;
}