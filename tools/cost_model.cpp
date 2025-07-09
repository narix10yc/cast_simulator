#include "cast/Core/CostModel.h"
#include "cast/CPU/CPUKernelManager.h"
#include "utils/iocolor.h"
#include "llvm/Support/CommandLine.h"

#include <fstream>
namespace cl = llvm::cl;

#define ERR_PRECISION 1
#define ERR_FILENAME 2
#define ERR_FILE_IO 3

cl::opt<std::string>
ArgOutputFilename("o", cl::desc("Output file name"), cl::Required);

cl::opt<bool>
ArgForceFilename("force-name",
  cl::desc("Force output file name as it is, possibly not ending with .csv"),
  cl::init(false));

cl::opt<bool>
ArgF32("f32", cl::desc("Enable single-precision"), cl::init(true));

cl::opt<bool>
ArgF64("f64", cl::desc("Enable double-precision"), cl::init(true));

cl::opt<int>
ArgNQubits("nqubits", cl::desc("Number of qubits"), cl::init(28));

cl::opt<int>
ArgNThreads("T", cl::desc("Number of threads"), cl::Prefix, cl::init(0));

cl::opt<bool>
ArgOverwriteMode("overwrite",
  cl::desc("Overwrite the output file with new results"),
  cl::init(false));

cl::opt<int>
ArgSimdWidth("simd-width", cl::desc("simd width"), cl::init(0));

cl::opt<int>
ArgNTests("N", cl::desc("Number of tests"), cl::Prefix, cl::Required);

using namespace cast;

bool checkFileName() {
  if (ArgForceFilename)
    return false;
  const std::string& fileName = ArgOutputFilename;
  if (fileName.length() > 4 && fileName.ends_with(".csv"))
    return false;

  std::cerr << BOLDYELLOW("Notice: ")
            << "Output filename does not end with '.csv'. "
               "If this filename is desired, please add '-force' "
               "commandline option\n";
  return true;
}

int unwrapNumThreads() {
  if (ArgNThreads == 0)
    return cast::get_cpu_num_threads();
  return ArgNThreads;
}

CPUSimdWidth unwrapSimdWidth() {
  if (ArgSimdWidth == 0)
    return get_cpu_simd_width();
  else 
    return static_cast<CPUSimdWidth>(static_cast<int>(ArgSimdWidth));
}

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv);
  Precision precision;
  if (checkPrecisionArgsCollision(precision))
    return ERR_PRECISION;
  
  if (checkFileName())
    return ERR_FILENAME;

  std::ifstream inFile;
  std::ofstream outFile;
  if (ArgOverwriteMode)
    outFile.open(ArgOutputFilename, std::ios::out | std::ios::trunc);
  else
    outFile.open(ArgOutputFilename, std::ios::app);

  inFile.open(ArgOutputFilename, std::ios::in);
  if (!outFile || !inFile) {
    std::cerr << BOLDRED("[Error]: ")
              << "Unable to open file '" << ArgOutputFilename << "'.\n";
    return ERR_FILE_IO;
  }

  // If the file is empty (new cost model), write the CSV title
  if (inFile.peek() == std::ifstream::traits_type::eof())
    outFile << PerformanceCache::CSV_Title << "\n";
  inFile.close();

  PerformanceCache cache;
  auto nThreads = unwrapNumThreads();
  auto simdWidth = unwrapSimdWidth();

  CPUKernelGenConfig cpuConfig(simdWidth, precision);
  cache.runExperiments(cpuConfig, ArgNQubits, nThreads, ArgNTests);
  cache.writeResults(outFile);

  outFile.close();
  return 0;
}

