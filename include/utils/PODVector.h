#ifndef UTILS_VECTOR_H
#define UTILS_VECTOR_H

#include "utils/is_pod.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>

namespace utils {

/// According to C++ standards, \c resize always default-initialize elements
/// even for POD data. This is a simple vector implementation that allows more
/// efficient POD vector handling.
template <is_pod T> class PODVector {
  T* _data;
  size_t _capacity;
  size_t _size;

  static inline T* _allocate(size_t s) {
    return static_cast<T*>(::operator new(
        sizeof(T) * s, static_cast<std::align_val_t>(alignof(T))));
  }

public:
  /// This constructor will NOT initialize array
  PODVector() { reserve(1); }

  explicit PODVector(size_t size)
      : _data(_allocate(size)), _capacity(size), _size(size) {}

  PODVector(size_t size, const T& value)
      : _data(_allocate(size)), _capacity(size), _size(size) {
    for (size_t i = 0; i < size; i++)
      _data[i] = value;
  }

  ~PODVector() { ::operator delete(_data); }

  PODVector(const PODVector& other)
      : _data(_allocate(other.capacity())), _capacity(other._capacity),
        _size(other._size) {
    ::memcpy(_data, other._data, other.sizeInBytes());
  }

  PODVector(PODVector&& other) noexcept
      : _data(other._data), _capacity(other._capacity), _size(other._size) {
    other._data = nullptr;
  }

  PODVector& operator=(const PODVector& other) {
    if (this == &other)
      return *this;
    this->~Vector();
    new (this) PODVector(other); // copy-construct the entire Vector
    return *this;
  }

  PODVector& operator=(PODVector&& other) noexcept {
    if (this == &other)
      return *this;
    this->~Vector();
    new (this) PODVector(std::move(other)); // move-construct the entire Vector
    return *this;
  }

  size_t size() const { return _size; }
  size_t capacity() const { return _capacity; }

  size_t sizeInBytes() const { return sizeof(T) * _size; }
  size_t capacityInBytes() const { return sizeof(T) * _capacity; }

  bool empty() const { return _size == 0; }

  T& operator[](size_t idx) {
    assert(idx < _size);
    return _data[idx];
  }

  const T& operator[](size_t idx) const {
    assert(idx < _size);
    return _data[idx];
  }

  T* begin() { return _data; }
  const T* begin() const { return _data; }
  const T* cbegin() const { return _data; }

  T* end() { return _data + _size; }
  const T* end() const { return _data + _size; }
  const T* cend() const { return _data + _size; }

  T& front() {
    assert(_size > 0);
    return _data[0];
  }
  const T& front() const {
    assert(_size > 0);
    return _data[0];
  }

  T& back() {
    assert(_size > 0);
    return _data[_size - 1];
  }
  const T& back() const {
    assert(_size > 0);
    return _data[_size - 1];
  }

  void push_back(const T& value) {
    if (_size == _capacity)
      reserve(_capacity << 1);
    _data[_size++] = value;
  }

  void push_back(T&& value) {
    if (_size == _capacity)
      reserve(_capacity << 1);
    _data[_size++] = std::move(value);
  }

  template <typename... Args> void emplace_back(Args&&... args) {
    if (_size == _capacity)
      reserve(_capacity << 1);
    new (_data + _size++) T(std::forward<Args>(args)...);
  }

  // Destroy the last element and reduce size by 1.
  void pop_back() {
    assert(_size > 0);
    --_size;
  }

  void reserve(size_t capacity) {
    if (capacity == 0)
      capacity = 1;
    if (capacity <= _capacity)
      return;
    T* newData = _allocate(capacity);
    ::memcpy(newData, _data, sizeInBytes());
    ::operator delete(_data);
    _data = newData;
    _capacity = capacity;
  }

  void resize(size_t size) {
    assert(size >= _size);
    reserve(size);
    _size = size;
  }
};

} // namespace utils

#endif // UTILS_VECTOR_H
