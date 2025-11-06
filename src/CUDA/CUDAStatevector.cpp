#include "cast/CUDA/CUDAStatevector.h"
#include "cast/CUDA/Config.h"
#include "utils/Formats.h"

#include <cassert>
#include <iostream>

using namespace cast;

template <typename ScalarType>
CUDAStatevector<ScalarType>::CUDAStatevector(const CUDAStatevector& other)
    : nQubits_(other.nQubits_), dData_(nullptr), hData_(nullptr) {
  // copy device data
  if (other.dData_ != nullptr) {
    mallocDeviceData();
    CUDA_CALL(
        cudaMemcpy(
            dData_, other.dData_, sizeInBytes(), cudaMemcpyDeviceToDevice),
        "Failed to copy array device to device");
  }
  // copy host data
  if (other.hData_ != nullptr) {
    mallocHostData();
    std::memcpy(hData_, other.hData_, sizeInBytes());
  }
}

template <typename ScalarType>
CUDAStatevector<ScalarType>::CUDAStatevector(CUDAStatevector&& other)
    : nQubits_(other.nQubits_), dData_(other.dData_), hData_(other.hData_) {
  other.dData_ = nullptr;
  other.hData_ = nullptr;
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
  assert(dData_ == nullptr && "Device data is already allocated");
  CUDA_CALL(cudaMalloc(&dData_, sizeInBytes()),
            "Failed to allocate device data");
  CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize device");
}

template <typename ScalarType>
void CUDAStatevector<ScalarType>::freeDeviceData() {
  assert(dData_ != nullptr &&
         "Device data is not allocated when trying to free it");
  CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize device");
  CUDA_CALL(cudaFree(dData_), "Failed to free device data");
  dData_ = nullptr;
}

template <typename ScalarType> void CUDAStatevector<ScalarType>::initialize() {
  if (dData_ == nullptr)
    mallocDeviceData();
  CUDA_CALL(cudaMemset(dData_, 0, sizeInBytes()),
            "Failed to zero statevector on the device");
  ScalarType one = 1.0;
  CUDA_CALL(
      cudaMemcpy(dData_, &one, sizeof(ScalarType), cudaMemcpyHostToDevice),
      "Failed to set the first element of the statevector to 1");
}

template <typename ScalarType>
ScalarType CUDAStatevector<ScalarType>::normSquared() const {
  using Helper = cast::internal::HelperCUDAKernels<ScalarType>;
  assert(dData_ != nullptr && "Device statevector is not initialized");
  return Helper::reduceSquared(dData_, size());
}

template <typename ScalarType> void CUDAStatevector<ScalarType>::randomize() {
  using Helper = cast::internal::HelperCUDAKernels<ScalarType>;
  if (dData_ == nullptr)
    mallocDeviceData();
  Helper::randomizeStatevector(dData_, size());

  // normalize the statevector
  auto c = 1.0 / norm();
  Helper::multiplyByConstant(dData_, c, size());
  cudaDeviceSynchronize();
}

template <typename ScalarType>
std::complex<ScalarType> CUDAStatevector<ScalarType>::amp(size_t idx) const {
  assert(dData_ != nullptr);
  assert(2ULL * idx + 1 < size());

  ScalarType re, im;
  CUDA_CHECK(cudaMemcpy(
      &re, dData_ + 2ULL * idx, sizeof(ScalarType), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&im,
                        dData_ + 2ULL * idx + 1,
                        sizeof(ScalarType),
                        cudaMemcpyDeviceToHost));
  return std::complex<ScalarType>(re, im);
}

template <typename ScalarType>
ScalarType CUDAStatevector<ScalarType>::prob(int qubit) const {
  using Helper = cast::internal::HelperCUDAKernels<ScalarType>;
  assert(dData_ != nullptr);

  return 1.0 - Helper::reduceSquaredOmittingBit(dData_, size(), qubit + 1);
}

template <typename ScalarType> void CUDAStatevector<ScalarType>::sync() const {
  CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize device");
  assert(dData_ != nullptr && "Device statevector is not initialized");

  if (hData_ == nullptr)
    mallocHostData();

  CUDA_CALL(cudaMemcpy(hData_, dData_, sizeInBytes(), cudaMemcpyDeviceToHost),
            "Failed to copy statevector from device to host");
  CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize device");
}

template <typename ScalarType>
std::ostream& CUDAStatevector<ScalarType>::display(std::ostream& os) const {
  sync();
  unsigned l = std::min(32U, 1U << nQubits_);
  for (unsigned i = 0; i < l; ++i)
    os << utils::fmt_0b(i, 5) << " : "
       << utils::fmt_complex(hData_[2 * i], hData_[2 * i + 1]) << "\n";
  return os;
}

namespace cast {
template class CUDAStatevector<float>;
template class CUDAStatevector<double>;
} // namespace cast
