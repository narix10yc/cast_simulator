#ifdef CAST_USE_CUDA

#include "cast/CUDA/CUDAStatevector.h"
#include "utils/iocolor.h"

#include <cassert>
#include <iostream>

using namespace cast;

template <typename ScalarType>
CUDAStatevector<ScalarType>::CUDAStatevector(const CUDAStatevector& other)
    : _nQubits(other._nQubits), _dData(nullptr), _hData(nullptr) {
  // copy device data
  if (other._dData != nullptr) {
    mallocDeviceData();
    CUDA_CALL(
        cudaMemcpy(
            _dData, other._dData, sizeInBytes(), cudaMemcpyDeviceToDevice),
        "Failed to copy array device to device");
  }
  // copy host data
  if (other._hData != nullptr) {
    mallocHostData();
    std::memcpy(_hData, other._hData, sizeInBytes());
  }
}

template <typename ScalarType>
CUDAStatevector<ScalarType>::CUDAStatevector(CUDAStatevector&& other)
    : _nQubits(other._nQubits), _dData(other._dData), _hData(other._hData) {
  other._dData = nullptr;
  other._hData = nullptr;
}

template <typename ScalarType>
CUDAStatevector<ScalarType>&
CUDAStatevector<ScalarType>::operator=(const CUDAStatevector& other) {
  if (this == &other)
    return *this;
  this->~CUDAStatevector();
  new (this) CUDAStatevector(other);
  return *this;
}

template <typename ScalarType>
CUDAStatevector<ScalarType>&
CUDAStatevector<ScalarType>::operator=(CUDAStatevector&& other) {
  if (this == &other)
    return *this;
  this->~CUDAStatevector();
  new (this) CUDAStatevector(std::move(other));
  return *this;
}

template <typename ScalarType>
void CUDAStatevector<ScalarType>::mallocDeviceData() {
  assert(_dData == nullptr && "Device data is already allocated");
  CUDA_CALL(cudaMalloc(&_dData, sizeInBytes()),
            "Failed to allocate device data");
  CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize device");
}

template <typename ScalarType>
void CUDAStatevector<ScalarType>::freeDeviceData() {
  assert(_dData != nullptr &&
         "Device data is not allocated when trying to free it");
  // For safety, we always synchronize the device before freeing memory
  CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize device");
  CUDA_CALL(cudaFree(_dData), "Failed to free device data");
  _dData = nullptr;
}

template <typename ScalarType> void CUDAStatevector<ScalarType>::initialize() {
  if (_dData == nullptr)
    mallocDeviceData();
  CUDA_CALL(cudaMemset(_dData, 0, sizeInBytes()),
            "Failed to zero statevector on the device");
  ScalarType one = 1.0;
  CUDA_CALL(
      cudaMemcpy(_dData, &one, sizeof(ScalarType), cudaMemcpyHostToDevice),
      "Failed to set the first element of the statevector to 1");
}

template <typename ScalarType>
ScalarType CUDAStatevector<ScalarType>::normSquared() const {
  using Helper = cast::internal::HelperCUDAKernels<ScalarType>;
  assert(_dData != nullptr && "Device statevector is not initialized");
  return Helper::reduceSquared(_dData, size());
}

template <typename ScalarType> void CUDAStatevector<ScalarType>::randomize() {
  using Helper = cast::internal::HelperCUDAKernels<ScalarType>;
  if (_dData == nullptr)
    mallocDeviceData();
  Helper::randomizeStatevector(_dData, size());

  // normalize the statevector
  auto c = 1.0 / norm();
  Helper::multiplyByConstant(_dData, c, size());
  cudaDeviceSynchronize();
}

template <typename ScalarType>
ScalarType CUDAStatevector<ScalarType>::prob(int qubit) const {
  using Helper = cast::internal::HelperCUDAKernels<ScalarType>;
  assert(_dData != nullptr);

  return 1.0 - Helper::reduceSquaredOmittingBit(_dData, size(), qubit + 1);
}

template <typename ScalarType> void CUDAStatevector<ScalarType>::sync() {
  CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize device");
  assert(_dData != nullptr &&
         "Device array is not initialized when calling sync()");
  // ensure host data is allocated
  if (_hData == nullptr)
    mallocHostData();

  CUDA_CALL(cudaMemcpy(_hData, _dData, sizeInBytes(), cudaMemcpyDeviceToHost),
            "Failed to copy statevector from device to host");
  CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize device");
}

namespace cast {
template class CUDAStatevector<float>;
template class CUDAStatevector<double>;
} // namespace cast

#endif // CAST_USE_CUDA