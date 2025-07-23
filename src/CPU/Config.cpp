#include "cast/CPU/Config.h"
#include "utils/iocolor.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <thread>

using namespace cast;

static int parsePositiveInt(const char* str) {
  if (str == nullptr)
    return 0;

  int value = 0;
  while (*str) {
    if (!std::isdigit(*str))
      return 0;
    value = value * 10 + (*str - '0');
    ++str;
  }
  return value;
}

int cast::get_cpu_num_threads() {
  const char* env = std::getenv("CAST_NUM_THREADS");
  int envNThreads = parsePositiveInt(env);
  if (envNThreads > 0)
    return envNThreads;

// CMAKE_CAST_NUM_THREADS is forwarded from CMake by CMake option
// -DCAST_NUM_THREADS=<N>
#ifdef CMAKE_CAST_NUM_THREADS
  if (CMAKE_CAST_NUM_THREADS > 0)
    return CMAKE_CAST_NUM_THREADS;
#endif // #ifdef CMAKE_CAST_NUM_THREADS

  int nThreads = std::thread::hardware_concurrency();
  if (nThreads <= 0) {
    std::cerr << BOLDYELLOW("[Warning]: ")
              << "Unable to detect hardware concurrency, defaulting to 1 "
                 "thread. "
                 "You can set the number of threads using the "
                 "CAST_NUM_THREADS "
                 "environment variable or by defining "
                 "-DCAST_NUM_THREADS=<N> "
                 "when configuring the project.\n";
    nThreads = 1;
  }
  return nThreads;
}

CPUSimdWidth cast::get_cpu_simd_width() {
  const char* env = std::getenv("CAST_SIMD_WIDTH");
  int width = parsePositiveInt(env);
  switch (width) {
  case (0):
    break;
  case (128):
    return W128;
  case (256):
    return W256;
  case (512):
    return W512;
  default:
    std::cerr << BOLDYELLOW("[Warning]: ")
              << "Environment variable CAST_SIMD_WIDTH is set to an illegal "
                 "value: "
              << width << ". Accepted values are: 128, 256, 512. Ignored.\n";
    break;
  }

  auto features = llvm::sys::getHostCPUFeatures();
  if (features.empty()) {
    std::cerr << BOLDYELLOW("[Warning]: ")
              << "llvm::sys::getHostCPUFeatures() returned empty features. "
                 "Defaulting SIMD width to 128 bits. "
                 "You may want to set environment variable CAST_SIMD_WIDTH "
                 "to provide an explicit value. Accepted values are: "
                 "128, 256, 512.\n";
    return W128;
  }
  if (features.lookup("avx512f"))
    return W512;
  if (features.lookup("avx2") || features.lookup("avx"))
    return W256;
  if (features.lookup("sse2") || features.lookup("sse") ||
      features.lookup("neon"))
    return W128;
  // If no SIMD features are detected, return W128 as a default.
  std::cerr << BOLDYELLOW("[Warning]: ")
            << "No SIMD features detected. Defaulting SIMD width to 128 bits."
               "You may want to set environment variable CAST_SIMD_WIDTH "
               "to provide an explicit value. Accepted values are: "
               "128, 256, 512.\n";
  return W128;
}

int cast::get_simd_s(CPUSimdWidth simdWidth, Precision precision) {
  if (precision == Precision::F32) {
    switch (simdWidth) {
    case W128:
      return 2; // 128 bits / 32 bits = 4 elements
    case W256:
      return 3; // 256 bits / 32 bits = 8 elements
    case W512:
      return 4; // 512 bits / 32 bits = 16 elements
    default:
      assert(false && "Unsupported SIMD Width");
      return 0;
    }
  }
  if (precision == Precision::F64) {
    switch (simdWidth) {
    case W128:
      return 1; // 128 bits / 64 bits = 2 elements
    case W256:
      return 2; // 256 bits / 64 bits = 4 elements
    case W512:
      return 3; // 512 bits / 64 bits = 8 elements
    default:
      assert(false && "Unsupported SIMD Width");
      return 0;
    }
  }
  assert(false && "Unsupported precision");
  return 0;
}