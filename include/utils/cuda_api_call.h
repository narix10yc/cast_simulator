#ifndef UTILS_CUDA_API_CALL_H
#define UTILS_CUDA_API_CALL_H

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


#endif // UTILS_CUDA_API_CALL_H