#ifndef CAST_CUDA_CONFIG_H
#define CAST_CUDA_CONFIG_H

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
#define CUDA_CHECK(CALL)                                                       \
  {                                                                            \
    cudaError_t err = CALL;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "\033[31m[CUDA Runtime Error]\033[0m "                      \
                << cudaGetErrorString(err) << " (at " << __FILE__ << ":"       \
                << __LINE__ << ")"                                             \
                << " in CALL: " << #CALL << "\n";                              \
    }                                                                          \
  }

// Driver API error checking
#define CU_CHECK(CALL)                                                         \
  {                                                                            \
    CUresult err = CALL;                                                       \
    if (err != CUDA_SUCCESS) {                                                 \
      const char* errStr;                                                      \
      cuGetErrorString(err, &errStr);                                          \
      std::cerr << "\033[31m[CUDA Driver Error]\033[0m " << errStr             \
                << " (code " << err << ")"                                     \
                << " at " << __FILE__ << ":" << __LINE__                       \
                << " in CALL: " << #CALL << "\n";                              \
    }                                                                          \
  }

namespace cast {

void displayCUDA();

void getCudaComputeCapability(int& major, int& minor);

} // namespace cast

#endif // CAST_CUDA_CONFIG_H