#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"

using namespace cast;

int main(int argc, char** argv) {
  CUDAStatevectorF32 sv(6);
  sv.randomize();

  return 0;
}