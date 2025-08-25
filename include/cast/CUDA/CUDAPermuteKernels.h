#ifndef CAST_CUDA_CUDAPERMUTEKERNELS_H
#define CAST_CUDA_CUDAPERMUTEKERNELS_H

#include "cast/CUDA/CUDALaunchPlan.h"
#include "cast/CUDA/CUDALayout.h"
#include <cstdint>
#include <span>

namespace cast {
void applyPermutationOnGPU(void* svReIm,
                           uint32_t nQubits,
                           std::span<const AxisSwap> swaps,
                           size_t sizeofScalar,
                           cudaStream_t stream);
}

#endif // CAST_CUDA_CUDAPERMUTEKERNELS_H
