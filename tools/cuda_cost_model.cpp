#include "cast/CUDA/CUDACostModel.h"
#include "cast/CUDA/CUDAKernelManager.h"

#include "utils/iocolor.h"
#include "llvm/Support/CommandLine.h"

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <set>
#include <string>

namespace cl = llvm::cl;
using namespace cast;

static cl::OptionCategory ArgCategory("CUDA Cost Model Options");

// clang-format off
static cl::opt<std::string>
ArgOutputFilename("o", cl::cat(ArgCategory),
    cl::desc("Output file name"),
    cl::Required);

static cl::opt<bool>
ArgOverwriteMode("overwrite", cl::cat(ArgCategory),
    cl::desc("Overwrite the output file with new results"),
    cl::init(false));

static cl::opt<int>
ArgNQubits("nqubits", cl::cat(ArgCategory),
    cl::desc("Number of qubits"), cl::init(28));

static cl::opt<int> 
ArgNWorkerThreads("worker-threads", cl::cat(ArgCategory),
    cl::desc("CPU threads used for JIT compilation (0 to auto-detect)"),
    cl::init(0));

static cl::opt<int>
ArgNTests("N", cl::cat(ArgCategory),
    cl::desc("Number of benchmarked kernels"),
    cl::Prefix,
    cl::Required);

static cl::opt<bool>
ArgF32("f32", cl::cat(ArgCategory),
    cl::desc("Enable single-precision (f32) kernels"),
    cl::init(false));

static cl::opt<bool>
ArgF64("f64", cl::cat(ArgCategory),
    cl::desc("Enable double-precision (f64) kernels"),
    cl::init(false));

static cl::opt<int>
ArgVerbose("verbose", cl::cat(ArgCategory),
    cl::desc("Verbosity level (0=quiet, 1=some, 2=more)"),
    cl::init(1));

// clang-format on

int main(int argc, char** argv) {
  cl::HideUnrelatedOptions({&ArgCategory});
  cl::ParseCommandLineOptions(argc, argv);

  // Unwrap arguments
  int nQubitsSV = ArgNQubits;
  int nWorkerThreads = (ArgNWorkerThreads <= 0) ? cast::get_cpu_num_threads()
                                                : ArgNWorkerThreads;
  if (!ArgF32 && !ArgF64) {
    std::cerr << BOLDRED("[Error]: ")
              << "At least one of -f32 or -f64 must be specified.\n";
    return 1;
  }

  std::ifstream inFile;
  std::ofstream outFile;
  if (ArgOverwriteMode)
    outFile.open(ArgOutputFilename, std::ios::out | std::ios::trunc);
  else
    outFile.open(ArgOutputFilename, std::ios::app);

  inFile.open(ArgOutputFilename, std::ios::in);
  if (!outFile || !inFile) {
    std::cerr << BOLDRED("[Error]: ") << "Unable to open file '"
              << ArgOutputFilename << "'.\n";
    return 1;
  }

  // If the file is empty (new cost model), write the CSV title
  if (inFile.peek() == std::ifstream::traits_type::eof())
    outFile << CUDAPerformanceCache::CSVTitle() << "\n";
  inFile.close();

  CUDAPerformanceCache cache;
  if (ArgF32) {
    std::cerr << BOLDCYAN("[Info]: ")
              << "Running single-precision experiments.\n";
    CUDAKernelGenConfig cudaConfig(Precision::F32);
    if (auto err = cache.runExperiments(
            cudaConfig, nQubitsSV, nWorkerThreads, ArgNTests)) {
      std::cerr << BOLDRED("[Error]: ") << "Failed to run f32 experiments: "
                << llvm::toString(std::move(err)) << "\n";
      return 1;
    }
  }
  if (ArgF64) {
    std::cerr << BOLDCYAN("[Info]: ")
              << "Running double-precision experiments.\n";
    CUDAKernelGenConfig cudaConfig(Precision::F64);
    if (auto err = cache.runExperiments(
            cudaConfig, nQubitsSV, nWorkerThreads, ArgNTests)) {
      std::cerr << BOLDRED("[Error]: ") << "Failed to run f64 experiments: "
                << llvm::toString(std::move(err)) << "\n";
      return 1;
    }
  }
  cache.writeResults(outFile);

  outFile.close();
  return 0;
}
