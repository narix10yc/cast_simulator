#include "cast/CUDA/CUDACostModel.h"
#include "utils/PrintSpan.h"
#include <numeric>

using namespace cast;

int main(int argc, char** argv) {
  CUDAPerformanceCache cache;
  CUDAKernelGenConfig config;
  llvm::cantFail(cache.runExperiments(config, 29, 4, 20, 2));

  return 0;
}