#ifndef CAST_CUDA_CONFIG_H
#define CAST_CUDA_CONFIG_H

#include <cstdint>
#include <cstdlib>  // For std::abort
#include <iostream> // For std::cerr

#ifndef CAST_DISABLE_ABORT_ON_CUDA_ERROR
// Driver API, abort on non-success values with custom error message
#define CU_CALL(FUNC, MSG)                                                     \
  do {                                                                         \
    if (auto cuResult = FUNC; cuResult != CUDA_SUCCESS) {                      \
      std::cerr << "\033[31m[CUDA Driver Error]\033[0m " << MSG << ". Func "   \
                << __PRETTY_FUNCTION__ << ". Error code " << cuResult << "\n"; \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

// Runtime API, abort on non-success values with custom error message
#define CUDA_CALL(FUNC, MSG)                                                   \
  do {                                                                         \
    if (auto cudaResult = FUNC; cudaResult != cudaSuccess) {                   \
      std::cerr << "\033[31m[CUDA Runtime Error]\033[0m " << MSG << ". Func "  \
                << __PRETTY_FUNCTION__                                         \
                << ". Error: " << cudaGetErrorString(cudaResult) << "\n";      \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

// Runtime API error checking, abort on non-success values
#define CUDA_CHECK(CALL)                                                       \
  do {                                                                         \
    cudaError_t logerr = CALL;                                                 \
    if (logerr != cudaSuccess) {                                               \
      std::cerr << "\033[31m[CUDA Runtime Error]\033[0m "                      \
                << cudaGetErrorString(logerr) << " (at " << __FILE__ << ":"    \
                << __LINE__ << ")" << " in CALL: " << #CALL << "\n";           \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

// Driver API error checking, abort on non-success values
#define CU_CHECK(CALL)                                                         \
  do {                                                                         \
    CUresult logerr = CALL;                                                    \
    if (logerr != CUDA_SUCCESS) {                                              \
      const char* errStr;                                                      \
      cuGetErrorString(logerr, &errStr);                                       \
      std::cerr << "\033[31m[CUDA Driver Error]\033[0m " << errStr             \
                << " (code " << logerr << ")" << " at " << __FILE__ << ":"     \
                << __LINE__ << " in CALL: " << #CALL << "\n";                  \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#else // CAST_DISABLE_ABORT_ON_CUDA_ERROR

// Driver API
#define CU_CALL(FUNC, MSG)                                                     \
  do {                                                                         \
    if (auto cuResult = FUNC; cuResult != CUDA_SUCCESS) {                      \
      std::cerr << "\033[31m[CUDA Driver Error]\033[0m " << MSG << ". Func "   \
                << __PRETTY_FUNCTION__ << ". Error code " << cuResult << "\n"; \
    }                                                                          \
  } while (0)

// Runtime API
#define CUDA_CALL(FUNC, MSG)                                                   \
  do {                                                                         \
    if (auto cudaResult = FUNC; cudaResult != cudaSuccess) {                   \
      std::cerr << "\033[31m[CUDA Runtime Error]\033[0m " << MSG << ". Func "  \
                << __PRETTY_FUNCTION__                                         \
                << ". Error: " << cudaGetErrorString(cudaResult) << "\n";      \
    }                                                                          \
  } while (0)

// Runtime API error checking
#define CUDA_CHECK(CALL)                                                       \
  do {                                                                         \
    cudaError_t logerr = CALL;                                                 \
    if (logerr != cudaSuccess) {                                               \
      std::cerr << "\033[31m[CUDA Runtime Error]\033[0m "                      \
                << cudaGetErrorString(logerr) << " (at " << __FILE__ << ":"    \
                << __LINE__ << ")" << " in CALL: " << #CALL << "\n";           \
    }                                                                          \
  } while (0)

// Driver API error checking
#define CU_CHECK(CALL)                                                         \
  do {                                                                         \
    CUresult logerr = CALL;                                                    \
    if (logerr != CUDA_SUCCESS) {                                              \
      const char* errStr;                                                      \
      cuGetErrorString(logerr, &errStr);                                       \
      std::cerr << "\033[31m[CUDA Driver Error]\033[0m " << errStr             \
                << " (code " << logerr << ")" << " at " << __FILE__ << ":"     \
                << __LINE__ << " in CALL: " << #CALL << "\n";                  \
    }                                                                          \
  } while (0)

#endif // CAST_DISABLE_ABORT_ON_CUDA_ERROR

namespace cast {

#ifdef CAST_USE_CUDA
static constexpr int kWarpSize = 32;
static constexpr int kWarpBits = 5;
static constexpr uint32_t kFullMask = 0xFFFFFFFFu;

#ifndef GPU_BACKEND_CUDA
#define GPU_BACKEND_CUDA 1
#endif
#endif

void displayCUDA();

void getCudaComputeCapability(int& major, int& minor);

} // namespace cast

#endif // CAST_CUDA_CONFIG_H