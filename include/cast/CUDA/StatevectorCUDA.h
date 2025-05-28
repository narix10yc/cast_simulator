#ifndef UTILS_STATEVECTOR_CUDA_H
#define UTILS_STATEVECTOR_CUDA_H

#include "utils/cuda_api_call.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring> // for std::memcpy
#include <cassert>
#include <cmath>

namespace utils {
  namespace internal {
    template<typename ScalarType>
    struct HelperCUDAKernels {
      static void multiplyByConstant(
        ScalarType* dArr, ScalarType constant, size_t size);

      // Randomize the \c dArr array using standard normal distribution
      static void randomizeStatevector(ScalarType* dArr, size_t size);

      // Return the sum of the squared values of \c dArr array
      static ScalarType reduceSquared(
          const ScalarType* dArr, size_t size);

      static ScalarType reduceSquaredOmittingBit(
          const ScalarType* dArr, size_t size, int bit);
    };

    extern template struct HelperCUDAKernels<float>;
    extern template struct HelperCUDAKernels<double>;
  } // namespace internal

/// @brief A helper class that relies on CUDA Runtime API to handle device data.
/// Notice that if using \c CUDAKernelManager, \c CUDAKernelManager instances 
/// must appear *before* any \c StatevectorCUDA instances, due to the order they
/// are destructed.
/// @tparam ScalarType 
template<typename ScalarType>
class StatevectorCUDA {
private:
  int _nQubits;
  // device data
  ScalarType* _dData;
  // host data
  ScalarType* _hData;

  // This function will call \c cudaDeviceSynchronize() after \c cudaMalloc()
  void mallocDeviceData();

  // This function will call \c cudaDeviceSynchronize() before \c cudaFree()
  void freeDeviceData();

  void mallocHostData() {
    assert(_hData == nullptr && "Host data is not null when trying malloc it");
    _hData = new ScalarType[size()];
  }

  void freeHostData() {
    assert(_hData != nullptr
      && "Host data is already null when trying to free it");
    delete[] _hData;
    _hData = nullptr;
  }
public:
  StatevectorCUDA(int nQubits)
  : _nQubits(nQubits), _dData(nullptr), _hData(nullptr) {}

  ~StatevectorCUDA() {
    if (_dData != nullptr)
      freeDeviceData();
    if (_hData != nullptr)
      freeHostData();
    assert(_dData == nullptr);
    assert(_hData == nullptr);
  }

  StatevectorCUDA(const StatevectorCUDA&);
  StatevectorCUDA(StatevectorCUDA&&);
  StatevectorCUDA& operator=(const StatevectorCUDA&);
  StatevectorCUDA& operator=(StatevectorCUDA&&);

  int nQubits() const { return _nQubits; }
  ScalarType* dData() const { return _dData; }
  ScalarType* hData() const { return _hData; }

  size_t sizeInBytes() const {
    return (2ULL << _nQubits) * sizeof(ScalarType);
  }

  // The size of statevector array, equaling to 2ULL << nQubits.
  size_t size() const { return 2ULL << _nQubits; }

  void initialize();

  ScalarType normSquared() const;
  ScalarType norm() const { return std::sqrt(normSquared()); }

  ScalarType prob(int qubits) const;

  void randomize();

  // This method will call \c cudaDeviceSynchronize(), copy device data (if 
  // exists) to host data, and call \c cudaDeviceSynchronize() again.
  void sync();

  void clearHostData() { if (_hData != nullptr) freeHostData(); }
};

extern template class StatevectorCUDA<float>;
extern template class StatevectorCUDA<double>;
} // namespace utils


#endif // UTILS_STATEVECTOR_CUDA_H
