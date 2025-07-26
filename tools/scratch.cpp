#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"

using namespace cast;

int main(int argc, char** argv) {
  CUDAStatevectorF32 sv(6);
  sv.randomize();
  std::cerr << (void*)sv.dData() << "\n";
  std::cerr << (void*)sv.hData() << "\n";

  return 0;
}