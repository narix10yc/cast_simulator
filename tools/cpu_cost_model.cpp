#include "cast/CPU/CPUCostModel.h"
#include "cast/CPU/CPUKernelManager.h"

#include "utils/iocolor.h"
#include "llvm/Support/CommandLine.h"

#include <fstream>
namespace cl = llvm::cl;
using namespace cast;

// clang-format off
cl::opt<std::string>
ArgOutputFilename("o", cl::desc("Output file name"), cl::Required);

cl::opt<bool>
ArgF32("f32", cl::desc("Enable single-precision"), cl::init(false));

cl::opt<bool>
ArgF64("f64", cl::desc("Enable double-precision"), cl::init(true));

cl::opt<int>
ArgNQubits("nqubits", cl::desc("Number of qubits"), cl::init(28));

cl::opt<int>
ArgNThreads("T", cl::desc("Number of threads"), cl::Prefix, cl::init(0));

cl::opt<bool>
ArgOverwriteMode("overwrite",
  cl::desc("Overwrite the output file with new results"), cl::init(false));

cl::opt<int>
ArgSimdWidth("simd-width", cl::desc("simd width"), cl::init(0));

cl::opt<int>
ArgNTests("N", cl::desc("Number of tests"), cl::Prefix, cl::Required);

// clang-format on

void unwrapArguments(int& nQubits, int& nThreads, CPUSimdWidth& simdWidth) {
  nQubits = ArgNQubits;
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
}

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv);

  int nQubits;
  int nThreads;
  CPUSimdWidth simdWidth;
  unwrapArguments(nQubits, nThreads, simdWidth);

  std::cerr << BOLDCYAN("[Info]: ")
            << "Building CPU cost model with the following parameters:\n"
            << "  - Number of qubits: " << nQubits << "\n"
            << "  - Number of threads: " << nThreads << "\n"
            << "  - SIMD width: " << static_cast<int>(simdWidth) << "\n"
            << "  - Output file: " << ArgOutputFilename << "\n";

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
    outFile << CPUPerformanceCache::Item::CSV_TITLE << "\n";
  inFile.close();

  CPUPerformanceCache cache;
  if (ArgF32) {
    std::cerr << BOLDCYAN("[Info]: ")
              << "Running single-precision experiments.\n";
    CPUKernelGenConfig cpuConfig(simdWidth, Precision::F32);
    cache.runExperiments(cpuConfig, ArgNQubits, nThreads, ArgNTests);
  }
  if (ArgF64) {
    std::cerr << BOLDCYAN("[Info]: ")
              << "Running double-precision experiments.\n";
    CPUKernelGenConfig cpuConfig(simdWidth, Precision::F64);
    cache.runExperiments(cpuConfig, ArgNQubits, nThreads, ArgNTests);
  }
  cache.writeResults(outFile);

  outFile.close();
  return 0;
}
