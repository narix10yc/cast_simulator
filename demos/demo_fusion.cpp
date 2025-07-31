#include "cast/CPU/CPUKernelManager.h"
#include "cast/CPU/CPUOptimizer.h"
#include "cast/CPU/CPUStatevector.h"

#include "timeit/timeit.h"
#include "utils/iocolor.h"

#include "llvm/Support/CommandLine.h"

namespace cl = llvm::cl;

using namespace cast;

// clang-format off
cl::opt<std::string>
ArgInputFilename("i",
  cl::desc("Input file name"), cl::Positional, cl::Required);

cl::opt<std::string>
ArgModelPath("model", cl::desc("Path to performance model"), cl::init(""));

cl::opt<int>
ArgPrecision("precision",
  cl::desc("Precision for the simulation (32 or 64)"), cl::init(64));

cl::opt<int>
ArgSimdWidth("simd-width",
  cl::desc("SIMD width (128, 256, 512, or 0 for auto-detect)"), cl::init(0));

cl::opt<int>
ArgNThreads("T", cl::desc("Number of threads"), cl::Prefix, cl::init(0));

cl::opt<bool>
ArgOverwriteMode("overwrite",
  cl::desc("Overwrite the output file with new results"), cl::init(false));

cl::opt<bool>
ArgRunNoFuse("run-no-fuse", cl::desc("Run no-fuse circuit"), cl::init(false));

cl::opt<bool>
ArgRunSizeOnlyFuse("run-sizeonly-fuse",
  cl::desc("Run size-only-fuse circuit"), cl::init(true));

cl::opt<bool>
ArgRunAdaptiveFuse("run-adaptive-fuse",
  cl::desc("Run adaptive-fuse circuit"), cl::init(false));

cl::opt<int>
ArgSizeonlySize("sizeonly-size",
  cl::desc("The max size of gates in size-only fusion"), cl::init(3));

cl::opt<bool>
ArgRunDenseKernel("run-dense-kernel",
  cl::desc("Run dense kernel"), cl::init(false));

cl::opt<int>
ArgReplication("replication", cl::desc("Number of replications"), cl::init(1));

cl::opt<int>
ArgVerbose("verbose", cl::desc("Verbosity level"), cl::init(1));

// clang-format on

static void unwrapArguments(Precision& precision,
                            int& nThreads,
                            CPUSimdWidth& simdWidth,
                            ir::CircuitNode& circuit) {
  if (ArgPrecision == 32)
    precision = Precision::F32;
  else if (ArgPrecision == 64)
    precision = Precision::F64;
  else {
    std::cerr << BOLDRED("[Error]: ")
              << "Invalid precision specified: " << ArgPrecision
              << ". Valid values are 32 or 64.\n";
    std::exit(1);
  }
  if (ArgNThreads <= 0)
    nThreads = cast::get_cpu_num_threads();
  else
    nThreads = ArgNThreads;

  if (ArgSimdWidth == 128)
    simdWidth = CPUSimdWidth::W128;
  else if (ArgSimdWidth == 256)
    simdWidth = CPUSimdWidth::W256;
  else if (ArgSimdWidth == 512)
    simdWidth = CPUSimdWidth::W512;
  else if (ArgSimdWidth == 0)
    simdWidth = get_cpu_simd_width();
  else {
    std::cerr << BOLDRED("[Error]: ")
              << "Invalid SIMD width specified: " << ArgSimdWidth
              << ". Valid values are 128, 256, 512, or 0 for auto-detection.\n";
    std::exit(1);
  }

  if (ArgRunAdaptiveFuse && ArgModelPath == "") {
    std::cerr
        << BOLDRED("[Err] ")
        << "--model=<path> must be specified when using adaptive fusion.\n";
    std::exit(1);
  }

  if (!(ArgRunNoFuse || ArgRunSizeOnlyFuse || ArgRunAdaptiveFuse)) {
    std::cerr << BOLDRED("[Err] ")
              << "At least one of --run-no-fuse, --run-naive-fuse, "
              << "--run-adaptive-fuse must be specified.\n";
    std::exit(1);
  }

  // parse source QASM file
  auto circuitOrErr = cast::parseCircuitFromQASMFile(ArgInputFilename);
  if (!circuitOrErr) {
    std::cerr << BOLDRED("[Err] ")
              << "Failed to parse circuit from file: " << ArgInputFilename
              << "\n"
              << "Error: " << circuitOrErr.takeError() << "\n";
    std::exit(1);
  }
  circuit = circuitOrErr.takeValue();
}

static void runAndDisplayResult(std::ostream& os,
                                CPUKernelManager& kernelMgr,
                                const std::string& graphName,
                                cast::CPUStatevectorWrapper& sv,
                                int nThreads,
                                bool isDense) {
  auto kernels = kernelMgr.getKernelsFromGraphName(graphName);
  double opCountTotal = 0.0;
  for (const auto& kernel : kernels)
    opCountTotal += kernel->gate->opCount(isDense ? 0.0 : 1e-8);

  timeit::Timer timer(ArgReplication);
  auto tr = timer.timeit([&]() {
    for (const auto& kernel : kernels) {
      kernelMgr.applyCPUKernel(sv.data(), sv.nQubits(), *kernel, nThreads)
          .consumeError();
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
}

int main(int argc, const char** argv) {
  cl::ParseCommandLineOptions(argc, argv);

  Precision precision;
  int nThreads;
  CPUSimdWidth simdWidth;
  ir::CircuitNode circuit("demo_circuit");
  unwrapArguments(precision, nThreads, simdWidth, circuit);

  ir::CircuitNode noFuseCircuit("no_fuse_circuit");
  circuit.body.deepcopyTo(noFuseCircuit.body);
  ir::CircuitNode sizeOnlyFuseCircuit("sizeonly_fuse_circuit");
  circuit.body.deepcopyTo(sizeOnlyFuseCircuit.body);
  ir::CircuitNode adaptiveFuseCircuit("adaptive_fuse_circuit");
  circuit.body.deepcopyTo(adaptiveFuseCircuit.body);

  ir::CircuitNode noFuseCircuitDense("no_fuse_dense_circuit");
  circuit.body.deepcopyTo(noFuseCircuitDense.body);
  ir::CircuitNode sizeOnlyFuseCircuitDense("sizeonly_fuse_dense_circuit");
  circuit.body.deepcopyTo(sizeOnlyFuseCircuitDense.body);
  ir::CircuitNode adaptiveFuseCircuitDense("adaptive_fuse_dense_circuit");
  circuit.body.deepcopyTo(adaptiveFuseCircuitDense.body);

  // Fusion
  utils::Logger logger(std::cerr, ArgVerbose);
  if (ArgRunSizeOnlyFuse) {
    CPUOptimizer opt;
    opt.setSizeOnlyFusionConfig(ArgSizeonlySize)
        .setNThreads(nThreads)
        .setPrecision(precision)
        .disableCFO();

    opt.run(sizeOnlyFuseCircuit, logger);
    if (ArgRunDenseKernel)
      opt.run(sizeOnlyFuseCircuitDense, logger);
  }

  if (ArgRunAdaptiveFuse) {
    auto pc = std::make_unique<CPUPerformanceCache>(
        static_cast<std::string>(ArgModelPath));

    auto cm = std::make_unique<CPUCostModel>(std::move(pc));
    cm->displayInfo(std::cerr, 2) << "\n";

    auto cpuFusionConfig =
        std::make_unique<CPUFusionConfig>(std::move(cm), nThreads, precision);

    CPUOptimizer opt;
    opt.setCPUFusionConfig(std::move(cpuFusionConfig))
        .setNThreads(nThreads)
        .setPrecision(precision)
        .disableCFO();

    opt.run(adaptiveFuseCircuit, logger);
    if (ArgRunDenseKernel)
      opt.run(adaptiveFuseCircuitDense, logger);
  }

  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.precision = precision;
  kernelGenConfig.simdWidth = simdWidth;
  // kernelGenConfig.matrixLoadMode = MatrixLoadMode::StackLoadMatElems;
  kernelGenConfig.displayInfo(std::cerr) << "\n";

  auto denseKernelGenConfig = kernelGenConfig;
  denseKernelGenConfig.zeroTol = 0.0;
  denseKernelGenConfig.oneTol = 0.0;

  // Generate kernels
  CPUKernelManager kernelMgr;
  if (ArgRunNoFuse) {
    utils::timedExecute(
        [&]() {
          auto r =
              kernelMgr.genGraphGates(kernelGenConfig,
                                      *noFuseCircuit.getAllCircuitGraphs()[0],
                                      "graphNoFuse");
          if (!r) {
            std::cerr << BOLDRED("[Err] ")
                      << "Failed to generate graphNoFuse\n";
            std::exit(1);
          }
        },
        "Generate No-fuse Kernels");
  }
  if (ArgRunSizeOnlyFuse) {
    utils::timedExecute(
        [&]() {
          auto r = kernelMgr.genGraphGates(
              kernelGenConfig,
              *sizeOnlyFuseCircuit.getAllCircuitGraphs()[0],
              "graphNaiveFuse");
          if (!r) {
            std::cerr << BOLDRED("[Err] ")
                      << "Failed to generate graphNaiveFuse\n";
            std::exit(1);
          }
        },
        "Generate Naive-fused Kernels");
  }
  if (ArgModelPath != "" && ArgRunAdaptiveFuse) {
    utils::timedExecute(
        [&]() {
          auto r = kernelMgr.genGraphGates(
              kernelGenConfig,
              *adaptiveFuseCircuit.getAllCircuitGraphs()[0],
              "graphAdaptiveFuse");
          if (!r) {
            std::cerr << BOLDRED("[Err] ")
                      << "Failed to generate graphAdaptiveFuse\n";
            std::exit(1);
          }
        },
        "Generate Adaptive-fused Kernels");
  }
  if (ArgRunNoFuse && ArgRunDenseKernel) {
    utils::timedExecute(
        [&]() {
          auto r =
              kernelMgr.genGraphGates(denseKernelGenConfig,
                                      *noFuseCircuit.getAllCircuitGraphs()[0],
                                      "graphNoFuseDense");
          if (!r) {
            std::cerr << BOLDRED("[Err] ")
                      << "Failed to generate graphNoFuseDense\n";
            std::exit(1);
          }
        },
        "Generate No-fuse Dense Kernels");
  }
  if (ArgRunSizeOnlyFuse && ArgRunDenseKernel) {
    utils::timedExecute(
        [&]() {
          auto r = kernelMgr.genGraphGates(
              denseKernelGenConfig,
              *sizeOnlyFuseCircuit.getAllCircuitGraphs()[0],
              "graphNaiveFuseDense");
          if (!r) {
            std::cerr << BOLDRED("[Err] ")
                      << "Failed to generate graphNaiveFuseDense\n";
            std::exit(1);
          }
        },
        "Generate Naive-fused Dense Kernels");
  }
  if (ArgModelPath != "" && ArgRunAdaptiveFuse && ArgRunDenseKernel) {
    utils::timedExecute(
        [&]() {
          auto r = kernelMgr.genGraphGates(
              denseKernelGenConfig,
              *adaptiveFuseCircuit.getAllCircuitGraphs()[0],
              "graphAdaptiveFuseDense");
          if (!r) {
            std::cerr << BOLDRED("[Err] ")
                      << "Failed to generate graphAdaptiveFuseDense\n";
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
        auto result = kernelMgr.initJIT(nThreads,
                                        llvm::OptimizationLevel::O1,
                                        /* useLazyJIT */ false,
                                        /* verbose */ 1);
        if (!result) {
          std::cerr << BOLDRED("[Err] ")
                    << "Failed to initialize JIT: " << result.takeError()
                    << "\n";
          std::exit(1);
        }
      },
      "JIT compile kernels");

  // Run kernels
  CPUStatevectorWrapper sv(
      precision, circuit.getAllCircuitGraphs()[0]->nQubits(), simdWidth);
  sv.randomize(nThreads);
  timeit::Timer timer(ArgReplication);
  timeit::TimingResult tr;

  std::cerr << BOLDCYAN("Running kernels:\n");
  if (ArgRunNoFuse) {
    runAndDisplayResult(std::cerr << "No-fuse Circuit:\n",
                        kernelMgr,
                        "graphNoFuse",
                        sv,
                        nThreads,
                        false);
    if (ArgRunDenseKernel) {
      runAndDisplayResult(std::cerr << "No-fuse Dense Circuit:\n",
                          kernelMgr,
                          "graphNoFuseDense",
                          sv,
                          nThreads,
                          true);
    }
  }

  if (ArgRunSizeOnlyFuse) {
    runAndDisplayResult(std::cerr << "Naive-fused Circuit:\n",
                        kernelMgr,
                        "graphNaiveFuse",
                        sv,
                        nThreads,
                        false);
    if (ArgRunDenseKernel) {
      runAndDisplayResult(std::cerr << "Naive-fused Dense Circuit:\n",
                          kernelMgr,
                          "graphNaiveFuseDense",
                          sv,
                          nThreads,
                          true);
    }
  }

  if (ArgRunAdaptiveFuse) {
    runAndDisplayResult(std::cerr << "Adaptive-fused Circuit:\n",
                        kernelMgr,
                        "graphAdaptiveFuse",
                        sv,
                        nThreads,
                        false);
    if (ArgRunDenseKernel) {
      runAndDisplayResult(std::cerr << "Adaptive-fused Dense Circuit:\n",
                          kernelMgr,
                          "graphAdaptiveFuseDense",
                          sv,
                          nThreads,
                          true);
    }
  }
  return 0;
}