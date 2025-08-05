#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDACostModel.h"

#include "utils/iocolor.h"
#include "llvm/Support/CommandLine.h"

#include <fstream>
#include <set>

namespace cl = llvm::cl;
using namespace cast;

cl::opt<std::string>
    ArgOutputFilename("o", cl::desc("Output file name"), cl::Required);

cl::opt<bool>
    ArgOverwriteMode("overwrite",
                     cl::desc("Overwrite the output file with new results"),
                     cl::init(false));

cl::opt<int> ArgNQubits("nqubits", cl::desc("Number of qubits"), cl::init(28));

cl::opt<int>
    ArgBlockSize("block-size",
                 cl::desc("CUDA block size (power of two ≤ 1024)"),
                 cl::init(256));

cl::opt<int>
    ArgWorkerThreads("worker-threads",
                     cl::desc("CPU threads used for JIT compilation"),
                     cl::init(0));

cl::opt<int>
    ArgNTests("N", cl::desc("Number of benchmarked kernels"),
              cl::Prefix, cl::Required);

cl::opt<bool>
    ArgF32("f32", cl::desc("Enable single-precision (f32) kernels"),
           cl::init(false));

cl::opt<bool>
    ArgF64("f64", cl::desc("Enable double-precision (f64) kernels"),
           cl::init(true));



static bool validateBlockSize(int blk) {
  if (blk < 32 || blk > 1024 || (blk & (blk - 1)) != 0) {
    std::cerr << BOLDRED("[Error]: ")
              << "Block size must be a power-of-two in [32, 1024].\n";
    return false;
  }
  return true;
}

static void ensureCsvHeader(std::ofstream &ofs) {
  ofs << CUDAPerformanceCache::Item::CSV_TITLE << '\n';
}


int main(int argc, char **argv)
{
  cl::ParseCommandLineOptions(argc, argv);

  if (!validateBlockSize(ArgBlockSize))
    return 1;

  std::ifstream inFile;
  std::ofstream outFile;

  if (ArgOverwriteMode)
    outFile.open(ArgOutputFilename, std::ios::out | std::ios::trunc);
  else
    outFile.open(ArgOutputFilename, std::ios::app);

  inFile.open(ArgOutputFilename, std::ios::in);
  if (!outFile || !inFile) {
    std::cerr << BOLDRED("[Error]: ")
              << "Unable to open '" << ArgOutputFilename << "'.\n";
    return 1;
  }
  if (inFile.peek() == std::ifstream::traits_type::eof())
    ensureCsvHeader(outFile);
  inFile.close();

  CUDAPerformanceCache cache;
  CUDAKernelGenConfig  cfg;
  cfg.blockSize = ArgBlockSize;

  std::cerr << BOLDCYAN("[Info]: ")
            << "Benchmarking " << ArgNTests << " kernels on "
            << ArgNQubits << " qubits – block " << ArgBlockSize
            << ", worker threads " << ArgWorkerThreads << ".\n";

  if (ArgF32) {
    std::cerr << BOLDCYAN("[Info]: ") << "  • Single precision (f32)\n";
    cfg.precision = Precision::F32;
    cache.runExperiments(cfg, ArgNQubits,
                         ArgBlockSize, ArgNTests, ArgWorkerThreads);
  }
  if (ArgF64) {
    std::cerr << BOLDCYAN("[Info]: ") << "  • Double precision (f64)\n";
    cfg.precision = Precision::F64;
    cache.runExperiments(cfg, ArgNQubits,
                         ArgBlockSize, ArgNTests, ArgWorkerThreads);
  }

  cache.writeResults(outFile);
  outFile.close();

  std::cerr << BOLDCYAN("[Info]: ")
            << "Results appended to '" << ArgOutputFilename << "'.\n";
  return 0;
}
