#ifndef CAST_CUDA_CUDAWARPCONFIG_H
#define CAST_CUDA_CUDAWARPCONFIG_H

#ifdef CAST_USE_CUDA
  static constexpr int kWarpSize = 32;
  static constexpr int kWarpBits = 5;
  using full_mask_t = unsigned; // 32-bit
  static constexpr full_mask_t kFullMask = 0xFFFFFFFFu;

  #ifndef GPU_BACKEND_CUDA
  # define GPU_BACKEND_CUDA 1
  #endif
#endif

#endif // CAST_CUDA_CUDAWARPCONFIG_H
