#include "cast/CUDA/CUDAPermute.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <vector>

__device__ __forceinline__ uint64_t deposit_by_mask(uint64_t x, uint64_t mask) {
  uint64_t res = 0, bb = 1;
  while (mask) {
    uint64_t bit = __ffsll(mask) - 1;
    if (x & bb)
      res |= (1ull << bit);
    bb <<= 1;
    mask &= (mask - 1);
  }
  return res;
}

template <typename Scalar, typename Vec2>
__global__ void permute_lowbits_kernel(const Scalar* __restrict__ src,
                                       Scalar* __restrict__ dst,
                                       uint64_t maskLow,
                                       int k,
                                       int nSys) {
  const uint64_t N = 1ull << nSys; // complex count
  const uint64_t FULL = (nSys == 64) ? ~0ull : ((1ull << nSys) - 1ull);
  const uint64_t maskHigh = FULL ^ maskLow;

  const Vec2* __restrict__ src2 = reinterpret_cast<const Vec2*>(src);
  Vec2* __restrict__ dst2 = reinterpret_cast<Vec2*>(dst);

  for (uint64_t j = blockIdx.x * blockDim.x + threadIdx.x; j < N;
       j += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t lo = (k ? (j & ((1ull << k) - 1ull)) : 0ull);
    const uint64_t hi = (k ? (j >> k) : j);
    const uint64_t i =
        deposit_by_mask(lo, maskLow) | deposit_by_mask(hi, maskHigh);
    dst2[j] = src2[i];
  }
}

extern "C" void cast_permute_lowbits_f32(const float* src,
                                         float* dst,
                                         int nSys,
                                         uint64_t maskLow,
                                         int k,
                                         cudaStream_t stream) {
  using V2 = float2;
  const uint64_t N = 1ull << nSys;
  dim3 block(256),
      grid((unsigned)std::min<uint64_t>((N + block.x - 1) / block.x, 65535));
  permute_lowbits_kernel<float, V2>
      <<<grid, block, 0, stream>>>(src, dst, maskLow, k, nSys);
}

extern "C" void cast_permute_lowbits_f64(const double* src,
                                         double* dst,
                                         int nSys,
                                         uint64_t maskLow,
                                         int k,
                                         cudaStream_t stream) {
  using V2 = double2;
  const uint64_t N = 1ull << nSys;
  dim3 block(256),
      grid((unsigned)std::min<uint64_t>((N + block.x - 1) / block.x, 65535));
  permute_lowbits_kernel<double, V2>
      <<<grid, block, 0, stream>>>(src, dst, maskLow, k, nSys);
}
