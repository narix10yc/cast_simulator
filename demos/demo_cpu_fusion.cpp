#include "cast/CPU/CPUKernelManager.h"
#include "cast/CPU/CPUOptimizer.h"
#include "cast/CPU/CPUStatevector.h"

#include "timeit/timeit.h"
#include "utils/PrintSpan.h"
#include "utils/iocolor.h"

#include "llvm/Support/CommandLine.h"

namespace cl = llvm::cl;

using namespace cast;

static cl::OptionCategory Category("Demo CPU Fusion Options");

// clang-format off
cl::opt<std::string>
ArgInputFilename("i", cl::cat(Category),
  cl::desc("Input file name"), cl::Positional, cl::Required);

cl::opt<std::string>
ArgModelPath("model", cl::cat(Category),
  cl::desc("Path to performance model"), cl::init(""));

cl::opt<int>
ArgPrecision("precision", cl::cat(Category),
  cl::desc("Precision for the simulation (32 or 64)"), cl::init(64));

cl::opt<int>
ArgSimdWidth("simd-width", cl::cat(Category),
  cl::desc("SIMD width (64, 128, 256, 512, or 0 for auto-detect)"), cl::init(0));

static cl::opt<std::string>
ArgOptMode("fusion", cl::cat(Category),
  cl::desc("Fusion optimization mode (mild, balanced, aggressive)"),
  cl::init("balanced"));

cl::opt<int>
ArgNWorkerThreads("T", cl::cat(Category),
  cl::desc("Number of threads"), cl::Prefix, cl::init(0));

cl::opt<bool>
ArgRunNoFuse("run-no-fuse", cl::cat(Category),
  cl::desc("Run no-fuse circuit"), cl::init(false));

cl::opt<bool>
ArgRunSizeOnlyFuse("run-sizeonly-fuse", cl::cat(Category),
  cl::desc("Run size-only-fuse circuit"), cl::init(true));

cl::opt<bool>
ArgRunAdaptiveFuse("run-adaptive-fuse", cl::cat(Category),
  cl::desc("Run adaptive-fuse circuit"), cl::init(false));

cl::opt<int>
ArgSizeonlySize("sizeonly-size", cl::cat(Category),
  cl::desc("The max size of gates in size-only fusion"), cl::init(3));

cl::opt<bool>
ArgRunDenseKernel("run-dense-kernel", cl::cat(Category),
  cl::desc("Run dense kernel"), cl::init(false));

cl::opt<int>
ArgReplication("replication", cl::cat(Category),
  cl::desc("Number of replications"), cl::init(1));

cl::opt<int>
ArgVerbose("verbose", cl::cat(Category),
  cl::desc("Verbosity level"), cl::init(1));

// clang-format on

static std::ostream& logerr() { return std::cerr << BOLDRED("[Error]: "); }

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

static CircuitGraphs unwrapArguments(Precision& precision,
                                     int& nThreads,
                                     CPUSimdWidth& simdWidth,
                                     int& nQubitsSV,
                                     FusionOptLevel& optLevel) {
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
    nThreads = cast::get_cpu_num_threads();
  else
    nThreads = ArgNWorkerThreads;

  if (ArgSimdWidth == 64)
    simdWidth = CPUSimdWidth::W64;
  else if (ArgSimdWidth == 128)
    simdWidth = CPUSimdWidth::W128;
  else if (ArgSimdWidth == 256)
    simdWidth = CPUSimdWidth::W256;
  else if (ArgSimdWidth == 512)
    simdWidth = CPUSimdWidth::W512;
  else if (ArgSimdWidth == 0)
    simdWidth = get_cpu_simd_width();
  else {
    logerr()
        << "Invalid SIMD width specified: " << ArgSimdWidth
        << ". Valid values are 64, 128, 256, 512, or 0 for auto-detection.\n";
    std::exit(1);
  }

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
  if (ArgOptMode == "mild")
    optLevel = FusionOptLevel::Mild;
  else if (ArgOptMode == "balanced")
    optLevel = FusionOptLevel::Balanced;
  else if (ArgOptMode == "aggressive")
    optLevel = FusionOptLevel::Aggressive;
  else {
    logerr() << "Unsupported fusion optimization mode: " << ArgOptMode
             << ". Supported modes are: mild, balanced, aggressive\n";
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

static void runAndDisplayResult(std::ostream& os,
                                CPUKernelManager& kernelMgr,
                                const std::string& graphName,
                                cast::CPUStatevectorWrapper& sv,
                                int nThreads,
                                bool isDense) {
  auto kernels = kernelMgr.getPool(graphName);
  double opCountTotal = 0.0;
  for (const auto& kernel : kernels)
    opCountTotal += kernel->gate->opCount(isDense ? 0.0 : 1e-8);

  timeit::Timer timer(ArgReplication);
  auto tr = timer.timeit([&]() {
    for (const auto& kernel : kernels) {
      if (auto e = kernelMgr.applyCPUKernel(
              sv.data(), sv.nQubits(), *kernel, nThreads)) {
        logerr() << "Failed to apply kernel " << kernel->llvmFuncName << ": "
                 << llvm::toString(std::move(e)) << "\n";
        std::exit(1);
      }
    }
  });

  double gflops =
      static_cast<double>(1ULL << sv.nQubits()) * 1e-9 * opCountTotal / tr.min;
  double bandwidth =
      static_cast<double>(sv.sizeInBytes()) * 1e-9 * kernels.size() / tr.min;
  os << "- Num Kernels:    " << kernels.size() << "\n"
     << "- Total Op Count: " << opCountTotal << "\n"
     << "- Fastest Run:    " << timeit::TimingResult::timeToString(tr.min, 4)
     << " @ " << gflops << " GFLOPs per second\n"
     << "- Effective Bandwidth: " << bandwidth << " GiBps\n";
  if (ArgVerbose > 1) {
    for (const auto& kernel : kernels) {
      os << kernel->llvmFuncName << ",\"";
      utils::printSpanNoBraket(os, std::span(kernel->gate->qubits()));
      os << "\"," << kernel->opCount << ","
         << utils::fmt_time(kernel->getExecTime()) << "\n";
    }
  }
}

int main(int argc, const char** argv) {
  cl::HideUnrelatedOptions(Category);
  cl::ParseCommandLineOptions(argc, argv);

  Precision precision;
  int nThreads;
  CPUSimdWidth simdWidth;
  int nQubitsSV;
  FusionOptLevel optLevel;
  auto graphs =
      unwrapArguments(precision, nThreads, simdWidth, nQubitsSV, optLevel);

  // Fusion
  utils::Logger logger(std::cerr, ArgVerbose);
  if (ArgRunSizeOnlyFuse) {
    CPUOptimizer opt;
    opt.setSizeOnlyFusionConfig(ArgSizeonlySize).enableCFO(false);
    opt.getFusionConfig()->setOptLevel(optLevel);

    opt.displayInfo({std::cerr, ArgVerbose});
    opt.run(graphs.sizeOnly, logger);
    if (ArgRunDenseKernel)
      opt.run(graphs.sizeOnlyDense, logger);
  }

  if (ArgRunAdaptiveFuse) {
    CPUOptimizer opt;
    opt.enableCFO(false);
    if (auto e = opt.loadCPUCostModel(ArgModelPath, nThreads, precision)) {
      logerr() << "Optimizer initialization failed: "
               << llvm::toString(std::move(e)) << "\n";
      std::exit(1);
    }
    opt.getFusionConfig()->setOptLevel(optLevel);

    opt.displayInfo({std::cerr, ArgVerbose});
    opt.run(graphs.adaptive, logger);
    if (ArgRunDenseKernel)
      opt.run(graphs.adaptiveDense, logger);
  }

  CPUKernelGenConfig kernelGenConfig(precision);
  kernelGenConfig.simdWidth = simdWidth;
  kernelGenConfig.displayInfo(std::cerr);
  std::cerr << "\n";

  auto denseKernelGenConfig = kernelGenConfig;
  denseKernelGenConfig.zeroTol = 0.0;
  denseKernelGenConfig.oneTol = 0.0;

  // Generate kernels
  CPUKernelManager km(nThreads);
  if (ArgRunNoFuse) {
    utils::timedExecute(
        [&]() {
          if (auto e = km.genGraphGates(
                  kernelGenConfig, graphs.noFuse, "graphNoFuse")) {
            logerr() << "Failed to generate graphNoFuse: "
                     << llvm::toString(std::move(e)) << "\n";
            std::exit(1);
          }
        },
        "Generate No-fuse Kernels");
  }
  if (ArgRunSizeOnlyFuse) {
    utils::timedExecute(
        [&]() {
          if (auto e = km.genGraphGates(
                  kernelGenConfig, graphs.sizeOnly, "graphNaiveFuse")) {
            logerr() << "Failed to generate graphNaiveFuse: "
                     << llvm::toString(std::move(e)) << "\n";
            std::exit(1);
          }
        },
        "Generate Naive-fused Kernels");
  }
  if (ArgModelPath != "" && ArgRunAdaptiveFuse) {
    utils::timedExecute(
        [&]() {
          if (auto e = km.genGraphGates(
                  kernelGenConfig, graphs.adaptive, "graphAdaptiveFuse")) {
            logerr() << "Failed to generate graphAdaptiveFuse: "
                     << llvm::toString(std::move(e)) << "\n";
            std::exit(1);
          }
        },
        "Generate Adaptive-fused Kernels");
  }
  if (ArgRunNoFuse && ArgRunDenseKernel) {
    utils::timedExecute(
        [&]() {
          if (auto e = km.genGraphGates(denseKernelGenConfig,
                                        graphs.noFuseDense,
                                        "graphNoFuseDense")) {
            logerr() << "Failed to generate graphNoFuseDense: "
                     << llvm::toString(std::move(e)) << "\n";
            std::exit(1);
          }
        },
        "Generate No-fuse Dense Kernels");
  }
  if (ArgRunSizeOnlyFuse && ArgRunDenseKernel) {
    utils::timedExecute(
        [&]() {
          if (auto e = km.genGraphGates(denseKernelGenConfig,
                                        graphs.sizeOnlyDense,
                                        "graphNaiveFuseDense")) {
            logerr() << "Failed to generate graphNaiveFuseDense: "
                     << llvm::toString(std::move(e)) << "\n";
            std::exit(1);
          }
        },
        "Generate Naive-fused Dense Kernels");
  }
  if (ArgModelPath != "" && ArgRunAdaptiveFuse && ArgRunDenseKernel) {
    utils::timedExecute(
        [&]() {
          if (auto e = km.genGraphGates(denseKernelGenConfig,
                                        graphs.adaptiveDense,
                                        "graphAdaptiveFuseDense")) {
            logerr() << "Failed to generate graphAdaptiveFuseDense: "
                     << llvm::toString(std::move(e)) << "\n";
            std::exit(1);
          }
        },
        "Generate Adaptive-fused Dense Kernels");
  }

  // JIT compile kernels
  std::vector<CPUKernelInfo*> kernelsNoFuse, kernelsNaiveFuse,
      kernelAdaptiveFuse;

  std::vector<CPUKernelInfo*> denseKernelsNoFuse, denseKernelsNaiveFuse,
      denseKernelAdaptiveFuse;

  utils::timedExecute(
      [&]() {
        if (auto e = km.compileAll(llvm::OptimizationLevel::O1, false, 1)) {
          logerr() << "Failed to initialize JIT: "
                   << llvm::toString(std::move(e)) << "\n";
          std::exit(1);
        }
      },
      "JIT compile kernels");

  // Run kernels
  CPUStatevectorWrapper sv(precision, nQubitsSV, simdWidth);
  sv.randomize(nThreads);
  timeit::Timer timer(ArgReplication);
  timeit::TimingResult tr;

  std::cerr << BOLDCYAN("Running kernels:\n");
  if (ArgRunNoFuse) {
    runAndDisplayResult(std::cerr << "No-fuse Circuit:\n",
                        km,
                        "graphNoFuse",
                        sv,
                        nThreads,
                        false);
    if (ArgRunDenseKernel) {
      runAndDisplayResult(std::cerr << "No-fuse Dense Circuit:\n",
                          km,
                          "graphNoFuseDense",
                          sv,
                          nThreads,
                          true);
    }
  }

  if (ArgRunSizeOnlyFuse) {
    runAndDisplayResult(std::cerr << "Naive-fused Circuit:\n",
                        km,
                        "graphNaiveFuse",
                        sv,
                        nThreads,
                        false);
    if (ArgRunDenseKernel) {
      runAndDisplayResult(std::cerr << "Naive-fused Dense Circuit:\n",
                          km,
                          "graphNaiveFuseDense",
                          sv,
                          nThreads,
                          true);
    }
  }

  if (ArgRunAdaptiveFuse) {
    runAndDisplayResult(std::cerr << "Adaptive-fused Circuit:\n",
                        km,
                        "graphAdaptiveFuse",
                        sv,
                        nThreads,
                        false);
    if (ArgRunDenseKernel) {
      runAndDisplayResult(std::cerr << "Adaptive-fused Dense Circuit:\n",
                          km,
                          "graphAdaptiveFuseDense",
                          sv,
                          nThreads,
                          true);
    }
  }
  return 0;
}