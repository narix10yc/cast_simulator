#ifndef CAST_CPU_CONFIG_H
#define CAST_CPU_CONFIG_H

#include "cast/Core/Precision.h"

namespace cast {
  enum CPUSimdWidth : int {
    W128 = 128, // 128 bits, such as SSE and NEON
    W256 = 256, // 256 bits, such as AVX and AVX2
    W512 = 512, // 512 bits, such as AVX-512
    W_Unknown = -1 // Unknown SIMD width
  };

  // Get the number of threads. Priority order:
  // 1. Environment variable CAST_NUM_THREADS
  // 2. -DCAST_NUM_THREADS=<N> during configuration
  // 3. Hardware concurrency (std::thread::hardware_concurrency())
  // 4. Default to 1 thread if all else fails
  int get_cpu_num_threads();

  // Get the native SIMD width in bits. This function relies on LLVM's 
  // implementation. If unavailable, we return 128 bits.
  CPUSimdWidth get_cpu_simd_width();

  // Get the simd_s to be used in kernel generation. Each SIMD register holds
  // (1 << simd_s) number of elements. For example, if simdWidth is W256,
  // then the 256-bit register can hold 4 double precision elements, so
  // simd_s will be 2 under F64 and 3 under F32.
  int get_simd_s(CPUSimdWidth simdWidth, Precision precision);

} // namespace cast

#endif // CAST_CPU_CONFIG_H