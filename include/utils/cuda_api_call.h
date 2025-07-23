#ifndef UTILS_CUDA_API_CALL_H
#define UTILS_CUDA_API_CALL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

// Driver API
#define CU_CALL(FUNC, MSG)                                                     \
  if (auto cuResult = FUNC; cuResult != CUDA_SUCCESS) {                        \
    std::cerr << "\033[31m[CUDA Driver Error]\033[0m " << MSG << ". Func "     \
              << __PRETTY_FUNCTION__ << ". Error code " << cuResult << "\n";   \
  }

// Runtime API
#define CUDA_CALL(FUNC, MSG)                                                   \
  if (auto cudaResult = FUNC; cudaResult != cudaSuccess) {                     \
    std::cerr << "\033[31m[CUDA Runtime Error]\033[0m " << MSG << ". Func "    \
              << __PRETTY_FUNCTION__                                           \
              << ". Error: " << cudaGetErrorString(cudaResult) << "\n";        \
  }

// Runtime API error checking
#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "\033[31m[CUDA Runtime Error]\033[0m "                      \
                << cudaGetErrorString(err) << " (at " << __FILE__ << ":"       \
                << __LINE__ << ")" << " in call: " << #call << "\n";           \
    }                                                                          \
  }

// Driver API error checking
#define CU_DRIVER_CHECK(call)                                                  \
  {                                                                            \
    CUresult err = call;                                                       \
    if (err != CUDA_SUCCESS) {                                                 \
      const char* errStr;                                                      \
      cuGetErrorString(err, &errStr);                                          \
      std::cerr << "\033[31m[CUDA Driver Error]\033[0m " << errStr             \
                << " (code " << err << ")" << " at " << __FILE__ << ":"        \
                << __LINE__ << " in call: " << #call << "\n";                  \
    }                                                                          \
  }

#endif // UTILS_CUDA_API_CALL_H