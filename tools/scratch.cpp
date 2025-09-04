#include "cast/CUDA/CUDACostModel.h"
#include "utils/PrintSpan.h"
#include <numeric>
#include <fstream>

using namespace cast;

int main(int argc, char** argv) {
  assert(argc == 2 && "Usage: ./cuda_cost_model <input_csv_file>");
  std::ifstream inFile(argv[1]);
  auto cacheExpected = CUDAPerformanceCache::LoadFrom(inFile);
  if (!cacheExpected) {
    std::cerr << "Error: Failed to load CUDAPerformanceCache from file '"
              << argv[1] << "': " << llvm::toString(cacheExpected.takeError())
              << "\n";
    return 1;
  }
  inFile.close();

  CUDACostModel cudaCM(cacheExpected.get());
  cudaCM.displayInfo(std::cerr, 2) << "\n";

  return 0;
}