#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAOptimizer.h"
#include "cast/CUDA/CUDAStatevector.h"

#include "timeit/timeit.h"
#include "utils/Formats.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

#include "llvm/Support/CommandLine.h"

namespace cl = llvm::cl;

using namespace cast;

#pragma region Command line arguments
// clang-format off

static cl::OptionCategory Category("Demo CUDA Fusion Options");

static cl::opt<std::string>
ArgInputFilename("i", cl::cat(Category),
  cl::desc("Input file name"), cl::Positional, cl::Required);

static cl::opt<std::string>
ArgModelPath("model", cl::cat(Category),
  cl::desc("Path to performance model"), cl::init(""));

static cl::opt<int>
ArgPrecision("precision", cl::cat(Category),
  cl::desc("Precision for the simulation (32 or 64)"), cl::init(64));

static cl::opt<int>
ArgNWorkerThreads("worker-threads", cl::cat(Category),
  cl::desc("Number of worker threads (0 to auto-detect)"), cl::init(0));

static cl::opt<bool>
ArgRunNoFuse("run-no-fuse", cl::cat(Category),
  cl::desc("Run no-fuse circuit"), cl::init(false));

static cl::opt<bool>
ArgRunSizeOnlyFuse("run-sizeonly-fuse", cl::cat(Category),
  cl::desc("Run size-only-fuse circuit"), cl::init(true));

static cl::opt<bool>
ArgRunAdaptiveFuse("run-adaptive-fuse", cl::cat(Category),
  cl::desc("Run adaptive-fuse circuit"), cl::init(false));

static cl::opt<int>
ArgSizeonlySize("sizeonly-size", cl::cat(Category),
  cl::desc("The max size of gates in size-only fusion"), cl::init(3));

static cl::opt<bool>
ArgRunDenseKernel("run-dense-kernel", cl::cat(Category),
  cl::desc("Run dense kernel"), cl::init(false));

static cl::opt<int>
ArgReplication("replication", cl::cat(Category),
  cl::desc("Number of replications"), cl::init(1));

static cl::opt<int>
ArgVerbose("verbose", cl::cat(Category),
  cl::desc("Verbosity level"), cl::init(1));

// clang-format on
#pragma endregion

static std::ostream& logerr() { return std::cerr << BOLDRED("[Err] "); }

struct CircuitGraphs {
  ir::CircuitGraphNode noFuse;
  ir::CircuitGraphNode sizeOnly;
  ir::CircuitGraphNode adaptive;

  ir::CircuitGraphNode noFuseDense;
  ir::CircuitGraphNode sizeOnlyDense;
  ir::CircuitGraphNode adaptiveDense;

  CircuitGraphs(const ir::CircuitGraphNode& cg) {
    auto allGates = cg.getAllGatesShared();
    for (const auto& gate : allGates) {
      noFuse.insertGate(gate);
      noFuseDense.insertGate(gate);
      sizeOnly.insertGate(gate);
      sizeOnlyDense.insertGate(gate);
      adaptive.insertGate(gate);
      adaptiveDense.insertGate(gate);
    }
  }
};

static CircuitGraphs
unwrapArguments(Precision& precision, int& nWorkerThreads, int& nQubitsSV) {
  if (ArgPrecision == 32)
    precision = Precision::FP32;
  else if (ArgPrecision == 64)
    precision = Precision::FP64;
  else {
    std::cerr << BOLDRED("[Error]: ")
              << "Invalid precision specified: " << ArgPrecision
              << ". Valid values are 32 or 64.\n";
    std::exit(1);
  }
  if (ArgNWorkerThreads <= 0)
    nWorkerThreads = cast::get_cpu_num_threads();
  else
    nWorkerThreads = ArgNWorkerThreads;

  if (ArgRunAdaptiveFuse && ArgModelPath == "") {
    logerr()
        << "--model=<path> must be specified when using adaptive fusion.\n";
    std::exit(1);
  }

  if (!(ArgRunNoFuse || ArgRunSizeOnlyFuse || ArgRunAdaptiveFuse)) {
    logerr() << "At least one of --run-no-fuse, --run-naive-fuse, "
             << "--run-adaptive-fuse must be specified.\n";
    std::exit(1);
  }

  // parse source QASM file
  auto circuit = cast::parseCircuitFromQASMFile(ArgInputFilename);
  if (!circuit) {
    logerr() << "Failed to parse circuit from file: " << ArgInputFilename
             << ": " << "Error: " << llvm::toString(circuit.takeError())
             << "\n";
    std::exit(1);
  }
  auto* cg = (*circuit)->getAllCircuitGraphs()[0];
  nQubitsSV = cg->nQubits();
  return CircuitGraphs(*cg);
}

int main(int argc, const char** argv) {
  cl::HideUnrelatedOptions(Category);
  cl::ParseCommandLineOptions(argc, argv);

  Precision precision;
  int nWorkerThreads;
  int nQubitsSV;
  auto graphs = unwrapArguments(precision, nWorkerThreads, nQubitsSV);

  /* Fusion */
  // Fusion: size only
  utils::Logger logger(std::cerr, ArgVerbose);
  if (ArgRunSizeOnlyFuse) {
    CUDAOptimizer opt;
    opt.setSizeOnlyFusionConfig(ArgSizeonlySize).enableCFO(false);

    opt.run(graphs.sizeOnly, logger);
    if (ArgRunDenseKernel)
      opt.run(graphs.sizeOnlyDense, logger);
  }

  // Fusion: adaptive
  if (ArgRunAdaptiveFuse) {
    CUDAOptimizer opt;
    opt.enableCFO(false);
    if (auto e = opt.loadCUDACostModelFromFile(ArgModelPath, precision)) {
      logerr() << "Failed to load CUDA cost model from " << ArgModelPath << ": "
               << llvm::toString(std::move(e)) << "\n";
      std::exit(1);
    }

    opt.run(graphs.adaptive, logger);
    if (ArgRunDenseKernel)
      opt.run(graphs.adaptiveDense, logger);
  }

  /* Kernel Generation */
  CUDAKernelGenConfig genCfg;
  genCfg.precision = precision;
  genCfg.displayInfo(std::cerr) << "\n";

  auto genDenseCfg = genCfg;
  genDenseCfg.zeroTol = 0.0;
  genDenseCfg.oneTol = 0.0;

  // Generate kernels
  CUDAKernelManager km(ArgNWorkerThreads);
  km.enableTiming();
  if (ArgRunNoFuse) {
    utils::timedExecute(
        [&]() {
          if (auto e = km.genGraphGates(genCfg, graphs.noFuse, "nofuse")) {
            logerr() << "Failed to generate nofuse graph: "
                     << llvm::toString(std::move(e)) << "\n";
            std::exit(1);
          }
        },
        "Generate No-fuse Kernels");
  }
  if (ArgRunSizeOnlyFuse) {
    utils::timedExecute(
        [&]() {
          if (auto e = km.genGraphGates(genCfg, graphs.sizeOnly, "sizeonly")) {
            logerr() << "Failed to generate size-only graph: "
                     << llvm::toString(std::move(e)) << "\n";
            std::exit(1);
          }
        },
        "Generate Size-only Fused Kernels");
  }
  if (ArgModelPath != "" && ArgRunAdaptiveFuse) {
    utils::timedExecute(
        [&]() {
          if (auto e = km.genGraphGates(genCfg, graphs.adaptive, "adaptive")) {
            logerr() << "Failed to generate adaptive graph: "
                     << llvm::toString(std::move(e)) << "\n";
            std::exit(1);
          }
        },
        "Generate Adaptive-fused Kernels");
  }
  if (ArgRunNoFuse && ArgRunDenseKernel) {
    utils::timedExecute(
        [&]() {
          if (auto e = km.genGraphGates(
                  genDenseCfg, graphs.noFuseDense, "nofuse_d")) {
            logerr() << "Failed to generate no-fuse dense graph: "
                     << llvm::toString(std::move(e)) << "\n";
            std::exit(1);
          }
        },
        "Generate No-fuse Dense Kernels");
  }
  if (ArgRunSizeOnlyFuse && ArgRunDenseKernel) {
    utils::timedExecute(
        [&]() {
          if (auto e = km.genGraphGates(
                  genDenseCfg, graphs.sizeOnlyDense, "sizeonly_d")) {
            logerr() << "Failed to generate size-only dense graph: "
                     << llvm::toString(std::move(e)) << "\n";
            std::exit(1);
          }
        },
        "Generate Size-only Fused Dense Kernels");
  }
  if (ArgModelPath != "" && ArgRunAdaptiveFuse && ArgRunDenseKernel) {
    utils::timedExecute(
        [&]() {
          if (auto e = km.genGraphGates(
                  genDenseCfg, graphs.adaptiveDense, "adaptive_d")) {
            logerr() << "Failed to generate adaptive-fuse dense graph: "
                     << llvm::toString(std::move(e)) << "\n";
            std::exit(1);
          }
        },
        "Generate Adaptive-fused Dense Kernels");
  }

  /* Launch and Time Kernels */
  CUDAStatevectorF64 sv(nQubitsSV);
  sv.randomize();
  km.setLaunchConfig(sv.getDevicePtr(), sv.nQubits());

  const auto runAndDisplayResult = [&](const std::string& graphName,
                                       bool isDense) {
    auto kernels = km.getKernelsFromGraphName(graphName);
    double opCountTotal = 0.0;
    for (const auto& kernel : kernels)
      opCountTotal += kernel->gate->opCount(isDense ? 0.0 : 1e-8);

    auto rawT0 = std::chrono::steady_clock::now();
    auto results = km.enqueueKernelLaunchFromGraph(graphName);
    km.syncKernelExecution();
    auto rawT1 = std::chrono::steady_clock::now();
    float t = 0.0f;
    for (const auto* r : results) {
      t += r->getKernelTime();
    }
    float rawT = std::chrono::duration<float>(rawT1 - rawT0).count();

    double gflops =
        static_cast<double>(1ULL << sv.nQubits()) * 1e-9 * opCountTotal;
    double bandwidth = kernels.size() *
                       static_cast<double>(1ULL << sv.nQubits()) *
                       (precision == Precision::FP32 ? 8.0 : 16.0) / t;
    std::cerr << "- Num Kernels:    " << kernels.size() << "\n"
              << "- Total Op Count: " << opCountTotal << "\n"
              << "- Run Time:       " 
              << timeit::TimingResult::timeToString(t, 4) << " @ " << gflops
              << " GFLOPs per second\n"
              << "- Wall Time:      "
              << timeit::TimingResult::timeToString(rawT, 4) << "\n"
              << "- Effective Bandwidth: " << utils::fmt_mem(bandwidth)
              << " per sec\n";
  };

  std::cerr << BOLDCYAN("Running kernels:\n");
  if (ArgRunNoFuse) {
    std::cerr << "No-fuse Circuit:\n";
    runAndDisplayResult("nofuse", false);
    if (ArgRunDenseKernel) {
      std::cerr << "No-fuse Dense Circuit:\n";
      runAndDisplayResult("nofuse_d", true);
    }
  }

  if (ArgRunSizeOnlyFuse) {
    std::cerr << "Size-only-fuse Circuit:\n";
    runAndDisplayResult("sizeonly", false);
    if (ArgRunDenseKernel) {
      std::cerr << "Size-only-fuse Dense Circuit:\n";
      runAndDisplayResult("sizeonly_d", true);
    }
  }

  if (ArgRunAdaptiveFuse) {
    std::cerr << "Adaptive-fused Circuit:\n";
    runAndDisplayResult("adaptive", false);
    if (ArgRunDenseKernel) {
      std::cerr << "Adaptive-fused Dense Circuit:\n";
      runAndDisplayResult("adaptive_d", true);
    }
  }
  return 0;
}