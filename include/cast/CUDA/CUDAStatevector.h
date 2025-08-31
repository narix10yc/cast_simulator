#ifndef CAST_CUDA_CUDASTATEVECTOR_H
#define CAST_CUDA_CUDASTATEVECTOR_H

#include "cast/CUDA/Config.h"

#include <cassert>
#include <cmath>
#include <cstring> // for std::memcpy
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

namespace cast {
namespace internal {
template <typename ScalarType> struct HelperCUDAKernels {
  static void
  multiplyByConstant(ScalarType* dArr, ScalarType constant, size_t size);

  // Randomize the \c dArr array using standard normal distribution
  static void randomizeStatevector(ScalarType* dArr, size_t size);

  // Return the sum of the squared values of \c dArr array
  static ScalarType reduceSquared(const ScalarType* dArr, size_t size);

  static ScalarType
  reduceSquaredOmittingBit(const ScalarType* dArr, size_t size, int bit);
};

extern template struct HelperCUDAKernels<float>;
extern template struct HelperCUDAKernels<double>;
} // namespace internal

/// @brief A helper class that relies on CUDA Runtime API to handle device data.
/// Notice that if using \c CUDAKernelManager, \c CUDAKernelManager instances
/// must appear *before* any \c CUDAStatevector instances, due to the order they
/// are destructed.
/// @tparam ScalarType
template <typename ScalarType> class CUDAStatevector {
private:
public:
  int nQubits_;
  // device data
  ScalarType* dData_;
  // host data
  mutable ScalarType* hData_;

  // This function will call \c cudaDeviceSynchronize() after \c cudaMalloc()
  void mallocDeviceData();

  // This function will call \c cudaDeviceSynchronize() before \c cudaFree()
  void freeDeviceData();

  void mallocHostData() const {
    assert(hData_ == nullptr && "Host data is not null when trying malloc it");
    hData_ = new ScalarType[size()];
  }

  void freeHostData() {
    assert(hData_ != nullptr &&
           "Host data is already null when trying to free it");
    delete[] hData_;
    hData_ = nullptr;
  }

public:
  CUDAStatevector(int nQubits, int deviceOrdinal = 0)
      : nQubits_(nQubits), dData_(nullptr), hData_(nullptr) {
    cudaSetDevice(deviceOrdinal);
    cudaFree(0);
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    std::cerr << "From CUDAStatevector: current context is " << ctx << "\n";
  }

  ~CUDAStatevector() {
    if (dData_ != nullptr)
      freeDeviceData();
    if (hData_ != nullptr)
      freeHostData();
    assert(dData_ == nullptr);
    assert(hData_ == nullptr);
  }

  CUDAStatevector(const CUDAStatevector&);
  CUDAStatevector(CUDAStatevector&&);
  CUDAStatevector& operator=(const CUDAStatevector&);
  CUDAStatevector& operator=(CUDAStatevector&&);

  int nQubits() const { return nQubits_; }
  ScalarType* dData() const { return dData_; }

  CUdeviceptr getDevicePtr() const {
    return reinterpret_cast<CUdeviceptr>(dData_);
  }
  ScalarType* hData() const { return hData_; }

  size_t sizeInBytes() const { return (2ULL << nQubits_) * sizeof(ScalarType); }

  // The size of statevector array, equaling to 2ULL << nQubits.
  size_t size() const { return 2ULL << nQubits_; }

  void initialize();

  void randomize();

  ScalarType normSquared() const;
  ScalarType norm() const { return std::sqrt(normSquared()); }

  ScalarType prob(int qubits) const;

  // This method will call \c cudaDeviceSynchronize(), copy device data (if
  // exists) to host data, and call \c cudaDeviceSynchronize() again.
  void sync() const;

  // Display the first few amplitudes in the statevector.
  std::ostream& display(std::ostream& os = std::cerr) const;

  void clearHostData() {
    if (hData_ != nullptr)
      freeHostData();
  }
};

extern template class CUDAStatevector<float>;
extern template class CUDAStatevector<double>;

using CUDAStatevectorF32 = CUDAStatevector<float>;
using CUDAStatevectorF64 = CUDAStatevector<double>;

} // namespace cast

#endif // CAST_CUDA_CUDASTATEVECTOR_H
