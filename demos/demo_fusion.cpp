#include "cast/CPU/CPUStatevector.h"

#include "cast/Fusion.h"
#include "cast/Transform/Transform.h"
#include "timeit/timeit.h"
#include "openqasm/parser.h"

#include "llvm/Support/CommandLine.h"

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
  cl::desc("Run naive-fuse circuit"), cl::init(true));

cl::opt<bool>
ArgRunAdaptiveFuse("run-adaptive-fuse",
  cl::desc("Run adaptive-fuse circuit"), cl::init(false));

cl::opt<int>
ArgNaiveMaxK("naive-max-k",
  cl::desc("The max size of gates in naive fusion"), cl::init(3));

cl::opt<bool>
ArgRunDenseKernel("run-dense-kernel",
  cl::desc("Run dense kernel"), cl::init(false));

cl::opt<int>
ArgReplication("replication",
  cl::desc("Number of replications"), cl::init(1));

static void runAndDisplayResult(
    CPUKernelManager& kernelMgr,
    const std::string& graphName,
    cast::CPUStatevector<double>& sv,
    bool denseKernel,
    std::ostream& os) {
  auto kernels = kernelMgr.collectKernelsFromGraphName(graphName);
  double opCountTotal = 0.0;
  for (const auto* kernel : kernels) {
    if (denseKernel)
      opCountTotal += static_cast<double>(1ULL << (kernel->gate->nQubits() + 2));
    else
      opCountTotal += kernel->gate->opCount(1e-8);
  }
  timeit::Timer timer(ArgReplication);
  auto tr = timer.timeit([&]() {
    for (auto* kernel : kernels) {
      kernelMgr.applyCPUKernelMultithread(
        sv.data(), sv.nQubits(), *kernel, ArgNThreads
      );
    }
  });
  double gflopsPerSec = static_cast<double>(1ULL << sv.nQubits()) * 1e-9 * opCountTotal / tr.min;
  double bandwidth = static_cast<double>(sv.sizeInBytes()) * 1e-9 * kernels.size() / tr.min;
  os << "- Num Kernels:    " << kernels.size() << "\n"
     << "- Total Op Count: " << opCountTotal << "\n"
     << "- Fastest Run:    " << timeit::TimingResult::timeToString(tr.min, 4) 
     << " @ " << gflopsPerSec << " GFLOPs per second\n"
     << "- Effective Bandwidth: " << bandwidth << " GiBps\n";
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

  // parse source QASM file
  openqasm::Parser qasmParser(ArgInputFilename, 0);
  auto qasmRoot = qasmParser.parse();
  ast::ASTContext astCtx;
  auto* circuitStmt = transform::cvtQasmCircuitToAstCircuit(*qasmRoot, astCtx);
  assert(circuitStmt != nullptr);
  auto irCircuit = transform::cvtAstCircuitToIrCircuit(*circuitStmt, astCtx);
  assert(irCircuit != nullptr);

  std::cerr << "AST Context: Allocated "
            << astCtx.bytesAllocated() << " bytes.\n";

  auto allGraphs = irCircuit->getAllCircuitGraphs();
  assert(allGraphs.size() == 1 && "There should be exactly one circuit graph.");

  const ir::CircuitGraphNode& graphOrginal = *allGraphs[0];
  ir::CircuitGraphNode graphNoFuse, graphNaiveFuse, graphAdaptiveFuse;
  ir::CircuitGraphNode graphNoFuseDense, graphNaiveFuseDense, graphAdaptiveFuseDense;

  if (ArgRunNoFuse) {
    graphNoFuse = graphOrginal;
    if (ArgRunDenseKernel) {
      graphNoFuseDense = graphOrginal;
    }
  }
  if (ArgRunNaiveFuse) {
    graphNaiveFuse = graphOrginal;
    if (ArgRunDenseKernel) {
      graphNaiveFuseDense = graphOrginal;
    }
  }
  if (ArgRunAdaptiveFuse) {
    graphAdaptiveFuse = graphOrginal;
    if (ArgRunDenseKernel) {
      graphAdaptiveFuseDense = graphOrginal;
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

  if (ArgRunAdaptiveFuse) {
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
  kernelGenConfig.matrixLoadMode = MatrixLoadMode::StackLoadMatElems;
  kernelGenConfig.displayInfo(std::cerr) << "\n";
  auto denseKernelGenConfig = kernelGenConfig;
  denseKernelGenConfig.zeroTol = 0.0;
  denseKernelGenConfig.oneTol = 0.0;
  denseKernelGenConfig.matrixLoadMode = MatrixLoadMode::StackLoadMatElems;

  // Generate kernels
  if (ArgRunNoFuse) {
    utils::timedExecute([&]() {
      kernelMgr.genCPUGatesFromGraph(
        kernelGenConfig, graphNoFuse, "graphNoFuse");
    }, "Generate No-fuse Kernels");
  }
  if (ArgRunNaiveFuse) {
    utils::timedExecute([&]() {
      kernelMgr.genCPUGatesFromGraph(
        kernelGenConfig, graphNaiveFuse, "graphNaiveFuse");
    }, "Generate Naive-fused Kernels");
  }
  if (ArgModelPath != "" && ArgRunAdaptiveFuse) {
    utils::timedExecute([&]() {
      kernelMgr.genCPUGatesFromGraph(
        kernelGenConfig, graphAdaptiveFuse, "graphAdaptiveFuse");
    }, "Generate Adaptive-fused Kernels");
  }
  if (ArgRunNoFuse && ArgRunDenseKernel) {
    utils::timedExecute([&]() {
      kernelMgr.genCPUGatesFromGraph(
        denseKernelGenConfig, graphNoFuseDense, "graphNoFuseDense");
    }, "Generate No-fuse Dense Kernels");
  }
  if (ArgRunNaiveFuse && ArgRunDenseKernel) {
    utils::timedExecute([&]() {
      kernelMgr.genCPUGatesFromGraph(
        denseKernelGenConfig, graphNaiveFuseDense, "graphNaiveFuseDense");
    }, "Generate Naive-fused Dense Kernels");
  }
  if (ArgModelPath != "" && ArgRunAdaptiveFuse && ArgRunDenseKernel) {
    utils::timedExecute([&]() {
      kernelMgr.genCPUGatesFromGraph(
        denseKernelGenConfig, graphAdaptiveFuseDense, "graphAdaptiveFuseDense");
    }, "Generate Adaptive-fused Dense Kernels");
  }

  // JIT compile kernels
  std::vector<CPUKernelInfo*>
  kernelsNoFuse, kernelsNaiveFuse, kernelAdaptiveFuse;

  std::vector<CPUKernelInfo*>
  denseKernelsNoFuse, denseKernelsNaiveFuse, denseKernelAdaptiveFuse;

  utils::timedExecute([&]() {
    kernelMgr.initJIT(
      ArgNThreads,
      llvm::OptimizationLevel::O1,
      /* useLazyJIT */ false,
      /* verbose */ 1
    );
  }, "JIT compile kernels");

  // Run kernels
  cast::CPUStatevector<double> sv(graphOrginal.nQubits(), kernelGenConfig.simd_s);
  // sv.randomize();
  timeit::Timer timer(ArgReplication);
  timeit::TimingResult tr;

  std::cerr << BOLDCYAN("Running kernels:\n");
  if (ArgRunNoFuse) {
    runAndDisplayResult(
      kernelMgr, "graphNoFuse", sv, false,
      std::cerr << "No-fuse Circuit:\n"
    );
    if (ArgRunDenseKernel) {
      runAndDisplayResult(
        kernelMgr, "graphNoFuseDense", sv, true,
        std::cerr << "No-fuse Dense Circuit:\n"
      );
    }
  }

  if (ArgRunNaiveFuse) {
    runAndDisplayResult(
      kernelMgr, "graphNaiveFuse", sv, false,
      std::cerr << "Naive-fused Circuit:\n"
    );
    if (ArgRunDenseKernel) {
      runAndDisplayResult(
        kernelMgr, "graphNaiveFuseDense", sv, true,
        std::cerr << "Naive-fused Dense Circuit:\n"
      );
    }
  }
  
  if (ArgRunAdaptiveFuse) {
    runAndDisplayResult(
      kernelMgr, "graphAdaptiveFuse", sv, false,
      std::cerr << "Adaptive-fused Circuit:\n"
    );
    if (ArgRunDenseKernel) {
      runAndDisplayResult(
        kernelMgr, "graphAdaptiveFuseDense", sv, true,
        std::cerr << "Adaptive-fused Dense Circuit:\n"
      );
    }
  }
  return 0;
}