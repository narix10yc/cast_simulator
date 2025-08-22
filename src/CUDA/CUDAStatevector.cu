#include "cast/CUDA/CUDAStatevector.h"
#include "cast/CUDA/CUDAWarpConfig.cuh"
#include <curand.h>
#include <curand_kernel.h>

using namespace cast;

template <typename ScalarType>
__global__ void
multiplyByConstantKernel(ScalarType* dArr, ScalarType c, size_t size) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size)
    dArr[idx] *= c;
}

template <typename ScalarType>
void cast::internal::HelperCUDAKernels<ScalarType>::multiplyByConstant(
    ScalarType* dArr, ScalarType c, size_t size) {
  size_t blockSize = 256;
  size_t numBlocks = (size + blockSize - 1) / blockSize;
  multiplyByConstantKernel<ScalarType><<<numBlocks, blockSize>>>(dArr, c, size);
}

template <typename ScalarType>
__global__ void initGaussianKernel(ScalarType* dArr,
                                   size_t size,
                                   ScalarType mean,
                                   ScalarType stddev) {
  size_t idx = threadIdx.x + (size_t)(blockIdx.x) * blockDim.x;
  if (idx < size) {
    curandState state;
    curand_init(0, idx, 0, &state);
    dArr[idx] = mean + stddev * curand_normal(&state);
  }
}

template <typename ScalarType>
void cast::internal::HelperCUDAKernels<ScalarType>::randomizeStatevector(
    ScalarType* dArr, size_t size) {
  size_t blockSize = 256;
  size_t numBlocks = (size + blockSize - 1) / blockSize;
  initGaussianKernel<ScalarType>
      <<<numBlocks, blockSize>>>(dArr, size, 0.0, 1.0);
}

template <typename ScalarType, unsigned blockSize>
__global__ void sumOfSquaredReductionKernel(const ScalarType* dArr,
                                            ScalarType* dOut,
                                            size_t size) {
  static_assert(blockSize == 32 || blockSize == 64 || blockSize == 128 ||
                blockSize == 256 || blockSize == 512);
  // __shared__ ScalarType shared[blockSize];
  // unsigned tid = threadIdx.x;
  __shared__ ScalarType shared[blockSize];
  const unsigned tid = threadIdx.x;
  const int lane         = threadIdx.x & (kWarpSize - 1);
  const int warpId       = threadIdx.x >> kWarpBits;
  const int warpsInBlock = blockDim.x >> kWarpBits;
  // Guard in case a generic kernel is re-used with a small block
  if (warpId >= warpsInBlock) return;
  unsigned bid = blockIdx.x;

  size_t i0 = (2ULL * bid) * blockSize + tid;
  size_t i1 = i0 + blockSize;
  shared[tid] = (i0 < size ? dArr[i0] * dArr[i0] : 0) +
                (i1 < size ? dArr[i1] * dArr[i1] : 0);
  __syncthreads();

  // Reduction in shared memory. The loop here sums the blockSize elements in
  // \c shared into shared[0].
  if constexpr (blockSize >= 512) {
    if (tid < 256)
      shared[tid] = shared[tid] + shared[tid + 256];
    __syncthreads();
  }
  if constexpr (blockSize >= 256) {
    if (tid < 128)
      shared[tid] = shared[tid] + shared[tid + 128];
    __syncthreads();
  }
  if constexpr (blockSize >= 128) {
    if (tid < 64)
      shared[tid] = shared[tid] + shared[tid + 64];
    __syncthreads();
  }
  // // threads in the same warp. No need to sync
  // if (tid < 32) {
  //   volatile ScalarType* vshared = shared;
  //   vshared[tid] = vshared[tid] + vshared[tid + 32];
  //   vshared[tid] = vshared[tid] + vshared[tid + 16];
  //   vshared[tid] = vshared[tid] + vshared[tid + 8];
  //   vshared[tid] = vshared[tid] + vshared[tid + 4];
  //   vshared[tid] = vshared[tid] + vshared[tid + 2];
  //   vshared[tid] = vshared[tid] + vshared[tid + 1];
  // }
  // Warp tail (first warp only). No sync needed within a warp.
  if (warpId == 0) {
    volatile ScalarType* vshared = shared;
    // Start at +kWarpSize if we arrived at 64-way partials, else at kWarpSize/2
    constexpr int firstOffset =
        (blockSize >= 2 * kWarpSize) ? kWarpSize : (kWarpSize >> 1);
    #pragma unroll
    for (int offset = firstOffset; offset > 0; offset >>= 1) {
      vshared[lane] = vshared[lane] + vshared[lane + offset];
    }
  }
  if (tid == 0)
    dOut[bid] = shared[0];
}

template <typename ScalarType>
ScalarType cast::internal::HelperCUDAKernels<ScalarType>::reduceSquared(
    const ScalarType* dArr, size_t size) {
  constexpr unsigned blockSize = 128;
  size_t gridSize = (size + 2 * blockSize - 1) / (2 * blockSize);

  ScalarType* dIntermediate;
  ScalarType* hIntermediate = new ScalarType[gridSize];

  cudaMalloc(&dIntermediate, gridSize * sizeof(ScalarType));

  // launch kernel
  sumOfSquaredReductionKernel<ScalarType, blockSize>
      <<<gridSize, blockSize>>>(dArr, dIntermediate, size);

  // final reduction on the host
  cudaMemcpy(hIntermediate,
             dIntermediate,
             gridSize * sizeof(ScalarType),
             cudaMemcpyDeviceToHost);
  ScalarType sum = 0;
  for (unsigned i = 0; i < gridSize; ++i)
    sum += hIntermediate[i];

  delete[] hIntermediate;
  cudaFree(dIntermediate);
  return sum;
}

template <typename ScalarType, unsigned blockSize>
__global__ void sumOfSquaredOmittingBitReductionKernel(const ScalarType* dArr,
                                                       ScalarType* dOut,
                                                       size_t size,
                                                       int bit) {
  static_assert(blockSize == 32 || blockSize == 64 || blockSize == 128 ||
                blockSize == 256 || blockSize == 512);
  // __shared__ ScalarType shared[blockSize];
  // unsigned tid = threadIdx.x;
  __shared__ ScalarType shared[blockSize];
  const unsigned tid = threadIdx.x;
  const int lane         = threadIdx.x & (kWarpSize - 1);
  const int warpId       = threadIdx.x >> kWarpBits;
  const int warpsInBlock = blockDim.x >> kWarpBits;
  // Guard in case a generic kernel is re-used with a small block
  if (warpId >= warpsInBlock) return;
  unsigned bid = blockIdx.x;

  size_t i0 = (2ULL * bid) * blockSize + tid;
  size_t i1 = i0 + blockSize;
  // insert a zero into bit-position 'bit'
  size_t mask = (1ULL << bit) - 1;
  i0 = ((i0 & ~mask) << 1) | (i0 & mask);
  i1 = ((i1 & ~mask) << 1) | (i1 & mask);
  shared[tid] = (i0 < size ? dArr[i0] * dArr[i0] : 0) +
                (i1 < size ? dArr[i1] * dArr[i1] : 0);
  __syncthreads();

  // Reduction in shared memory. The loop here sums the blockSize elements in
  // \c shared into shared[0].
  if constexpr (blockSize >= 512) {
    if (tid < 256)
      shared[tid] = shared[tid] + shared[tid + 256];
    __syncthreads();
  }
  if constexpr (blockSize >= 256) {
    if (tid < 128)
      shared[tid] = shared[tid] + shared[tid + 128];
    __syncthreads();
  }
  if constexpr (blockSize >= 128) {
    if (tid < 64)
      shared[tid] = shared[tid] + shared[tid + 64];
    __syncthreads();
  }
  // // threads in the same warp. No need to sync
  // if (tid < 32) {
  //   volatile ScalarType* vshared = shared;
  //   vshared[tid] = vshared[tid] + vshared[tid + 32];
  //   vshared[tid] = vshared[tid] + vshared[tid + 16];
  //   vshared[tid] = vshared[tid] + vshared[tid + 8];
  //   vshared[tid] = vshared[tid] + vshared[tid + 4];
  //   vshared[tid] = vshared[tid] + vshared[tid + 2];
  //   vshared[tid] = vshared[tid] + vshared[tid + 1];
  // }
  // Warp tail (first warp only). No sync needed within a warp.
  if (warpId == 0) {
    volatile ScalarType* vshared = shared;
    constexpr int firstOffset =
        (blockSize >= 2 * kWarpSize) ? kWarpSize : (kWarpSize >> 1);
    #pragma unroll
    for (int offset = firstOffset; offset > 0; offset >>= 1) {
      vshared[lane] = vshared[lane] + vshared[lane + offset];
    }
  }
  if (tid == 0)
    dOut[bid] = shared[0];
}

template <typename ScalarType>
ScalarType
cast::internal::HelperCUDAKernels<ScalarType>::reduceSquaredOmittingBit(
    const ScalarType* dArr, size_t size, int bit) {
  constexpr unsigned blockSize = 128;
  size_t gridSize = (size + 2 * blockSize - 1) / (2 * blockSize);

  ScalarType* dIntermediate;
  ScalarType* hIntermediate = new ScalarType[gridSize];

  cudaMalloc(&dIntermediate, gridSize * sizeof(ScalarType));

  // launch kernel
  sumOfSquaredOmittingBitReductionKernel<ScalarType, blockSize>
      <<<gridSize, blockSize>>>(dArr, dIntermediate, size, bit);

  // final reduction on the host
  cudaMemcpy(hIntermediate,
             dIntermediate,
             gridSize * sizeof(ScalarType),
             cudaMemcpyDeviceToHost);

  ScalarType sum = 0;
  for (unsigned i = 0; i < gridSize; ++i)
    sum += hIntermediate[i];

  delete[] hIntermediate;
  cudaFree(dIntermediate);
  return sum;
}

namespace cast::internal {
template struct HelperCUDAKernels<float>;
template struct HelperCUDAKernels<double>;
} // namespace cast::internal
