#include "cast/CPU/CPUKernelManager.h"
#include "cast/CPU/CPUOptimizer.h"
#include "cast/CPU/CPUStatevector.h"

#include "llvm/Support/CommandLine.h"

namespace cl = llvm::cl;

using namespace cast;

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

static cl::opt<std::string>
ArgOptMode("fusion", cl::cat(Category),
  cl::desc("Fusion optimization mode (mild, balanced, aggresive)"),
  cl::init("balanced"));

static cl::opt<int>
ArgVerbose("verbose", cl::cat(Category),
  cl::desc("Verbosity level"), cl::init(1));

// clang-format on

static Precision getPrecision() {
  if (ArgPrecision == 32)
    return Precision::FP32;
  else if (ArgPrecision == 64)
    return Precision::FP64;
  else {
    std::cerr << "Unsupported precision: " << ArgPrecision << "\n";
    exit(1);
  }
}

static FusionOptLevel getFusionOptLevel() {
  if (ArgOptMode == "mild")
    return FusionOptLevel::Mild;
  else if (ArgOptMode == "balanced")
    return FusionOptLevel::Balanced;
  else if (ArgOptMode == "aggressive")
    return FusionOptLevel::Aggressive;
  else {
    std::cerr << "Unsupported fusion optimization mode: " << ArgOptMode
              << ". Supported modes are: mild, balanced, aggressive\n";
    exit(1);
  }
}

using time_point = std::chrono::time_point<std::chrono::steady_clock>;
static auto getTime(time_point& a, time_point& b) {
  return utils::fmt_time(std::chrono::duration<float>(b - a).count());
}

int main(int argc, char** arv) {
  cl::HideUnrelatedOptions(Category);
  cl::ParseCommandLineOptions(argc, arv);
  auto precision = getPrecision();
  auto nThreads = cast::get_cpu_num_threads();

  using my_clock = std::chrono::steady_clock;

  auto t0 = my_clock::now();

  auto circuit = parseCircuitFromQASMFile(ArgInputFilename);
  if (!circuit) {
    std::cerr << "Error parsing circuit from " << ArgInputFilename << ": "
              << llvm::toString(circuit.takeError()) << "\n";
    return 1;
  }

  CPUOptimizer opt;
  if (auto e = opt.loadCPUCostModel(
          ArgModelPath, cast::get_cpu_num_threads(), precision)) {
    std::cerr << "Error loading cost model: " << llvm::toString(std::move(e))
              << "\n";
    return 1;
  }

  auto optLevel = getFusionOptLevel();
  opt.getFusionConfig()->setOptLevel(optLevel);
  opt.enableCFO(false);
  opt.run(**circuit);
  auto cg = (*circuit)->getAllCircuitGraphs()[0];

  // Right after optimization
  auto t1 = my_clock::now();

  CPUKernelManager km;
  CPUKernelGenConfig gConfig(precision);
  if (auto e = km.genGraphGates(gConfig, *cg, "graph")) {
    std::cerr << "Error generating kernels: " << llvm::toString(std::move(e))
              << "\n";
    return 1;
  }
  // Right after kernel generation
  auto t2 = my_clock::now();

  CPUStatevectorF64 sv(cg->nQubits(), gConfig.simdWidth);
  sv.initialize(nThreads);

  // Right after state init
  auto t3 = my_clock::now();
  if (auto e = km.initJIT()) {
    std::cerr << "Error initializing JIT: " << llvm::toString(std::move(e))
              << "\n";
    return 1;
  }

  // Right after JIT
  auto t4 = my_clock::now();

  if (auto e = km.applyCPUKernelsFromGraph(
          sv.data(), sv.nQubits(), "graph", nThreads)) {
    std::cerr << "Error applying kernels: " << llvm::toString(std::move(e))
              << "\n";
    return 1;
  }

  // Right after execution
  auto t5 = my_clock::now();

  // Statistics
  utils::InfoLogger logger(std::cerr, ArgVerbose);
  auto loggerB = logger.indent();
  logger.put("Input file", ArgInputFilename)
      .put("Num Threads", nThreads)
      .put("Precision", precision)
      .put("Num Gates After Opt", cg->nGates())
      .put("Total Time", getTime(t0, t5));
  loggerB.put("Parse & Optimize", getTime(t0, t1))
      .put("Kernel Gen", getTime(t1, t2))
      .put("State Init", getTime(t2, t3))
      .put("JIT", getTime(t3, t4))
      .put("Execution", getTime(t4, t5));
  return 0;
}