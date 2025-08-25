#include "cast/CUDA/CUDALayout.h"
#include "cast/CUDA/CUDAPermuteKernels.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void
swap_two_axes(T* __restrict svReIm, uint32_t n, uint32_t p, uint32_t q) {
  if (p == q)
    return;
  if (p > q) {
    uint32_t t = p;
    p = q;
    q = t;
  }
  const uint64_t dim = (1ull << n);
  const uint64_t maskP = (1ull << p);
  const uint64_t maskQ = (1ull << q);

  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dim;
       i += (uint64_t)gridDim.x * blockDim.x) {
    if (((i & maskP) == 0) && ((i & maskQ) != 0)) {
      uint64_t j = (i | maskP) & ~maskQ;
      uint64_t a = 2 * i;
      uint64_t b = 2 * j;
      T t0 = svReIm[a + 0];
      T t1 = svReIm[a + 1];
      svReIm[a + 0] = svReIm[b + 0];
      svReIm[a + 1] = svReIm[b + 1];
      svReIm[b + 0] = t0;
      svReIm[b + 1] = t1;
    }
  }
}

namespace cast {
void applyPermutationOnGPU(void* svReIm,
                           uint32_t nQubits,
                           std::span<const AxisSwap> swaps,
                           size_t sizeofScalar,
                           cudaStream_t stream) {
  if (swaps.empty())
    return;
  constexpr int TB = 256;
  uint64_t dim = 1ull << nQubits;
  int blocks = (int)std::min<uint64_t>((dim + TB - 1) / TB, 65535ull);
  for (auto [a, b] : swaps) {
    if (sizeofScalar == 4)
      swap_two_axes<float>
          <<<blocks, TB, 0, stream>>>((float*)svReIm, nQubits, a, b);
    else
      swap_two_axes<double>
          <<<blocks, TB, 0, stream>>>((double*)svReIm, nQubits, a, b);
  }
}
} // namespace cast
