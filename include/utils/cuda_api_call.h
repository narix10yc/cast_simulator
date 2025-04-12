#ifndef UTILS_CUDA_API_CALL_H
#define UTILS_CUDA_API_CALL_H

#ifndef RED
#define RED(x) "\033[31m" x "\033[0m"
#endif

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

// Driver API
#define CU_CALL(FUNC, MSG) \
  if (auto cuResult = FUNC; cuResult != CUDA_SUCCESS) { \
    std::cerr << RED("[CUDA Driver Error] ") \
              << MSG << ". Func " << __PRETTY_FUNCTION__ \
              << ". Error code " << cuResult << "\n"; \
  }

// Runtime API
#define CUDA_CALL(FUNC, MSG) \
  if (auto cudaResult = FUNC; cudaResult != cudaSuccess) { \
    std::cerr << RED("[CUDA Runtime Error] ") \
              << MSG << ". Func " << __PRETTY_FUNCTION__ \
              << ". Error: " << cudaGetErrorString(cudaResult) << "\n"; \
  }

// Runtime API error checking
#define CUDA_CHECK(call) {                                                \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
        std::cerr << RED("[CUDA Runtime Error] ")                         \
                  << cudaGetErrorString(err)                              \
                  << " (at " << __FILE__ << ":" << __LINE__ << ")"        \
                  << " in call: " << #call << "\n";                       \
    }                                                                     \
}

// Driver API error checking
#define CU_DRIVER_CHECK(call) {                                           \
    CUresult err = call;                                                  \
    if (err != CUDA_SUCCESS) {                                            \
        const char* errStr;                                               \
        cuGetErrorString(err, &errStr);                                   \
        std::cerr << RED("[CUDA Driver Error] ")                          \
                  << errStr << " (code " << err << ")"                    \
                  << " at " << __FILE__ << ":" << __LINE__                \
                  << " in call: " << #call << "\n";                       \
    }                                                                     \
}

#endif // UTILS_CUDA_API_CALL_H