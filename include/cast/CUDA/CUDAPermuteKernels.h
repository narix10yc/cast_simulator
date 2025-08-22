#pragma once
#include <cstdint>
#include <span>
#include "cast/CUDA/CUDALayout.h"
#include "cast/CUDA/CUDALaunchPlan.h"

namespace cast {
void applyPermutationOnGPU(void* svReIm,
                           uint32_t nQubits,
                           std::span<const AxisSwap> swaps,
                           size_t sizeofScalar,
                           cudaStream_t stream);
}
