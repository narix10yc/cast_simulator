#include "cast/CostModel.h"
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

cl::opt<int>
ArgPrecision("precision", cl::desc("Precision"), cl::init(64));

cl::opt<bool>
ArgF32("f32", cl::Optional, cl::init(false));

cl::opt<bool>
ArgF64("f64", cl::Optional, cl::init(false));

// return true on error
static bool checkPrecisionArgsCollision(int& precision) {
  if (ArgF32 && ArgF64) {
    std::cerr << BOLDRED("[Error]: ")
              << "-f32 and -f64 cannot be set together.\n";
    return true;
  }
  if (ArgF32) {
    if (ArgPrecision == 64) {
      std::cerr << BOLDRED("[Error]: ")
                << "Precision arguments contradict with each other.\n";
      return true;
    }
    precision = 32;
    return false;
  }
  if (ArgF64) {
    if (ArgPrecision == 32) {
      std::cerr << BOLDRED("[Error]: ")
                << "Precision arguments contradict with each other.\n";
      return true;
    }
    precision = 64;
    return false;
  }
  if (ArgPrecision != 32 && ArgPrecision != 64) {
    std::cerr << BOLDRED("[Error]: ")
              << "Precision should be either 32 or 64.\n";
    return true;
  }
  precision = ArgPrecision;
  return false;
}

cl::opt<int>
ArgNQubits("nqubits", cl::desc("Number of qubits"), cl::init(28));

cl::opt<int>
ArgNThreads("T", cl::desc("Number of threads"), cl::Prefix, cl::Required);

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

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv);
  int precision;
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
  CPUSimdWidth simdWidth;
  if (ArgSimdWidth == 0) {
    simdWidth = get_cpu_simd_width();
  else 
    simdWidth = static_cast<CPUSimdWidth>(static_cast<int>(ArgSimdWidth));
    
  CPUKernelGenConfig cpuConfig(simdWidth, precision);
  cache.runExperiments(cpuConfig, ArgNQubits, ArgNThreads, ArgNTests);
  cache.writeResults(outFile);

  outFile.close();
  return 0;
}

