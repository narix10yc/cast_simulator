#ifndef UTILS_PODARRAY_H
#define UTILS_PODARRAY_H

#include "utils/is_pod.h"
#include <cassert>
#include <cstdlib>

namespace utils {

template <is_pod T, size_t N> class PODArray {
  static_assert(N > 0, "N must be positive");
  T _data[N];

public:
  PODArray() = default;
  ~PODArray() = default;
  PODArray(const PODArray&) = default;
  PODArray(PODArray&&) = default;
  PODArray& operator=(const PODArray&) = default;
  PODArray& operator=(PODArray&&) = default;

  T* data() { return _data; }
  const T* data() const { return _data; }

  T* begin() { return _data; }
  T* end() { return _data + N; }
  const T* begin() const { return _data; }
  const T* end() const { return _data + N; }
  const T* cbegin() const { return _data; }
  const T* cend() const { return _data + N; }

  static constexpr size_t size() { return N; }
  static constexpr bool empty() { return N == 0; }

  T& operator[](size_t idx) {
    assert(idx < N);
    return _data[idx];
  }
  const T& operator[](size_t idx) const {
    assert(idx < N);
    return _data[idx];
  }
};

} // namespace utils

#endif // UTILS_PODARRAY_H
