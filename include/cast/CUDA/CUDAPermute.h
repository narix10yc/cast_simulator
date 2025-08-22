#pragma once
#ifdef CAST_USE_CUDA
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

extern "C" void cast_permute_lowbits_f32(const float* src, float* dst,
                                         int nSys, uint64_t maskLow, int k,
                                         cudaStream_t stream);
extern "C" void cast_permute_lowbits_f64(const double* src, double* dst,
                                         int nSys, uint64_t maskLow, int k,
                                         cudaStream_t stream);

template<typename T>
inline void cast_permute_lowbits(const T* src, T* dst, int nSys,
                                 uint64_t maskLow, int k,
                                 cudaStream_t stream = 0);

template<>
inline void cast_permute_lowbits<float>(const float* src, float* dst, int nSys,
                                        uint64_t maskLow, int k,
                                        cudaStream_t stream) {
  cast_permute_lowbits_f32(src, dst, nSys, maskLow, k, stream);
}

template<>
inline void cast_permute_lowbits<double>(const double* src, double* dst, int nSys,
                                         uint64_t maskLow, int k,
                                         cudaStream_t stream) {
  cast_permute_lowbits_f64(src, dst, nSys, maskLow, k, stream);
}

#endif
