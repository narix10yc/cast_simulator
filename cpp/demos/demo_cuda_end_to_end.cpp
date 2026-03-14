#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAOptimizer.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "utils/Formats.h"

#include <llvm/Support/CommandLine.h>

namespace cl = llvm::cl;

using namespace cast;

// clang-format off
static cl::OptionCategory Category("Demo CUDA Fusion Options");

static cl::opt<std::string>
ArgInputFilename("i", cl::cat(Category),
  cl::desc("Input file name"),
  cl::Positional,
  cl::Required);

static cl::opt<std::string>
ArgModelPath("model", cl::cat(Category),
  cl::desc("Path to performance model"),
  cl::Required);

static cl::opt<int>
ArgPrecision("precision", cl::cat(Category),
  cl::desc("Precision for the simulation (32 or 64)"),
  cl::init(64));

static cl::opt<int>
ArgVerbose("verbose", cl::cat(Category),
  cl::desc("Verbosity level"),
  cl::init(1));

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

using time_point = std::chrono::time_point<std::chrono::steady_clock>;
static auto getTime(time_point& a, time_point& b) {
  return utils::fmt_time(std::chrono::duration<float>(b - a).count());
}

int main(int argc, const char** argv) {
  llvm::cl::HideUnrelatedOptions(Category);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();

  auto circuit = cast::parseCircuitFromQASMFile(ArgInputFilename);
  if (!circuit) {
    llvm::errs() << "Error parsing QASM file: "
                 << llvm::toString(circuit.takeError()) << "\n";
    return 1;
  }
  CUDAOptimizer opt;
  auto precision = getPrecision();
  if (auto e = opt.loadCUDACostModelFromFile(ArgModelPath, precision)) {
    llvm::errs() << "Error loading CUDA cost model: "
                 << llvm::toString(std::move(e)) << "\n";
    return 1;
  }
  opt.enableCFO(false);
  opt.run(**circuit);

  // Right after parsing the circuit
  auto t1 = clock::now();

  auto* cg = (*circuit)->getAllCircuitGraphs()[0];
  CUDAStatevectorFP64 sv(cg->nQubits());
  sv.initialize();
  CUDAKernelManager km;
  km.enableTiming();
  km.setLaunchConfig(sv.getDevicePtr(), sv.nQubits());
  CUDAKernelGenConfig gConfig(precision);
  if (auto e = km.genGraphGates(gConfig, *cg, "graph")) {
    llvm::errs() << "Error generating CUDA kernels: "
                 << llvm::toString(std::move(e)) << "\n";
    return 1;
  }

  // Right after generated all kernels
  auto t2 = clock::now();

  km.enqueueKernelLaunchesFromGraph("graph");
  km.syncKernelExecution();

  // Right after all kernels finish execution
  auto t3 = clock::now();

  // Statistics
  utils::InfoLogger logger(std::cerr, ArgVerbose);
  auto loggerB = logger.indent();
  logger.put("Input QASM file", ArgInputFilename);
  logger.put("Precision", (precision == Precision::FP32 ? "32" : "64"));
  logger.put("Number of qubits", cg->nQubits());
  logger.put("Number of gates", cg->nGates());

  logger.put("End-to-end time (s)", getTime(t0, t3));
  loggerB.put("Parsing time (s)", getTime(t0, t1))
      .put("Kernel generation time (s)", getTime(t1, t2))
      .put("Kernel execution wall time (s)", getTime(t2, t3))
      .put("Kernel execution GPU time (s)",
           utils::fmt_time(km.getTotalExecTime()));

  return 0;
}