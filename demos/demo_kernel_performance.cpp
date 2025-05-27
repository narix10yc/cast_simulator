#include "simulation/KernelManager.h"

#include "llvm/Support/CommandLine.h"
namespace cl = llvm::cl;

using namespace cast;


int main(int argc, const char** argv) {
  cl::ParseCommandLineOptions(argc, argv);

  CPUKernelManager cpuKernelMgr;
  CPUKernelGenConfig cpuKernelGenConfig;
  cpuKernelGenConfig.displayInfo(std::cerr) << "\n";

}