#ifdef CAST_USE_CUDA

#include "simulation/StatevectorCUDA.h"
#include "utils/iocolor.h"

#include <cassert>
#include <iostream>

using namespace utils;

template<typename ScalarType>
StatevectorCUDA<ScalarType>::StatevectorCUDA(const StatevectorCUDA& other)
: _nQubits(other._nQubits), _dData(nullptr), _hData(nullptr) {
  // copy device data
  if (other._dData != nullptr) {
    mallocDeviceData();
    CUDA_CALL(
      cudaMemcpy(_dData, other._dData, sizeInBytes(), cudaMemcpyDeviceToDevice),
      "Failed to copy array device to device");
  }
  // copy host data
  if (other._hData != nullptr) {
    mallocHostData();
    std::memcpy(_hData, other._hData, sizeInBytes());
  }
}

template<typename ScalarType>
StatevectorCUDA<ScalarType>::StatevectorCUDA(StatevectorCUDA&& other) 
: _nQubits(other._nQubits), _dData(other._dData), _hData(other._hData) {
  other._dData = nullptr;
  other._hData = nullptr;
}

template<typename ScalarType>
StatevectorCUDA<ScalarType>&
StatevectorCUDA<ScalarType>::operator=(const StatevectorCUDA& other) {
  if (this == &other)
    return *this;
  this->~StatevectorCUDA();
  new (this) StatevectorCUDA(other);
  return *this;
}

template<typename ScalarType>
StatevectorCUDA<ScalarType>&
StatevectorCUDA<ScalarType>::operator=(StatevectorCUDA&& other) {
  if (this == &other)
    return *this;
  this->~StatevectorCUDA();
  new (this) StatevectorCUDA(std::move(other));
  return *this;
}

template<typename ScalarType>
void StatevectorCUDA<ScalarType>::mallocDeviceData() {
  assert(_dData == nullptr && "Device data is already allocated");
  CUDA_CALL(cudaMalloc(&_dData, sizeInBytes()),
    "Failed to allocate device data");
  CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize device");
}

template<typename ScalarType>
void StatevectorCUDA<ScalarType>::freeDeviceData() {
  assert(_dData != nullptr &&
    "Device data is not allocated when trying to free it");
  // For safety, we always synchronize the device before freeing memory
  CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize device");
  CUDA_CALL(cudaFree(_dData), "Failed to free device data");
  _dData = nullptr;
}

template<typename ScalarType>
void StatevectorCUDA<ScalarType>::initialize() {
  if (_dData == nullptr)
    mallocDeviceData();
  CUDA_CALL(cudaMemset(_dData, 0, sizeInBytes()),
    "Failed to zero statevector on the device");
  ScalarType one = 1.0;
  CUDA_CALL(cudaMemcpy(_dData, &one, sizeof(ScalarType), cudaMemcpyHostToDevice),
    "Failed to set the first element of the statevector to 1");
}

template<typename ScalarType>
ScalarType StatevectorCUDA<ScalarType>::normSquared() const {
  using Helper = utils::internal::HelperCUDAKernels<ScalarType>;
  assert(_dData != nullptr && "Device statevector is not initialized");
  return Helper::reduceSquared(_dData, size());
}

template<typename ScalarType>
void StatevectorCUDA<ScalarType>::randomize() {
  using Helper = utils::internal::HelperCUDAKernels<ScalarType>;
  if (_dData == nullptr)
    mallocDeviceData();
  Helper::randomizeStatevector(_dData, size());

  // normalize the statevector
  auto c = 1.0 / norm();
  Helper::multiplyByConstant(_dData, c, size());
  cudaDeviceSynchronize();
}

template<typename ScalarType>
ScalarType StatevectorCUDA<ScalarType>::prob(int qubit) const {
  using Helper = utils::internal::HelperCUDAKernels<ScalarType>;
  assert(_dData != nullptr);
 
  return 1.0 - Helper::reduceSquaredOmittingBit(_dData, size(), qubit + 1);
}

template<typename ScalarType>
void StatevectorCUDA<ScalarType>::sync() {
  CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize device");
  assert(_dData != nullptr &&
    "Device array is not initialized when calling sync()");
  // ensure host data is allocated
  if (_hData == nullptr)
    mallocHostData();

  CUDA_CALL(
    cudaMemcpy(_hData, _dData, sizeInBytes(), cudaMemcpyDeviceToHost),
    "Failed to copy statevector from device to host");
  CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize device");
}

namespace utils {
  template class StatevectorCUDA<float>;
  template class StatevectorCUDA<double>;
}

#endif // CAST_USE_CUDA