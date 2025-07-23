#ifndef UTILS_PODVARIANT_H
#define UTILS_PODVARIANT_H

#include "utils/is_pod.h"
#include <algorithm> // for std::max
#include <cassert>
#include <memory>

namespace utils {

struct Monostate {};

/// A lightweight variant class limited to POD types. \code get<T>()\endcode is
/// implemented simply as a cast and provides no run-time type-safety check
/// (except for in debug mode).
/// The intended example use is
/// \code
/// PODVariant<int, float> v1(1);
/// if (v1.is<int>()) {
///   auto i = v1.get<int>();
///   // use i safely
/// }
/// \endcode
template <is_pod... Types> class PODVariant {
  static constexpr size_t StorageSize = std::max({sizeof(Types)...});
  static constexpr size_t StorageAlign = std::max({alignof(Types)...});

  template <typename T> static constexpr int indexOf() {
    if constexpr (std::is_same_v<T, Monostate>)
      return -1;
    int index = 0;
    static_assert((std::is_same_v<const T&, const Types&> || ...),
                  "Wrong type");
    ((std::is_same_v<const T&, const Types&> ? 0 : ++index) && ...);
    return index;
  }

  alignas(StorageAlign) std::byte _storage[StorageSize];

  int _typeIndex;

public:
  /// Default constructor leaves the object storage in POD state
  PODVariant() : _typeIndex(-1) {}

  template <typename T> PODVariant(const T& value) {
    static_assert(indexOf<T>() < sizeof...(Types), "Wrong type");
    reinterpret_cast<T&>(_storage) = value;
    _typeIndex = indexOf<T>();
  }

  ~PODVariant() = default;
  PODVariant(const PODVariant&) = default;
  PODVariant(PODVariant&&) = default;
  PODVariant& operator=(const PODVariant&) = default;
  PODVariant& operator=(PODVariant&&) = default;

  template <typename T> PODVariant& operator=(const T& value) {
    static_assert(indexOf<T>() < sizeof...(Types), "Wrong type");
    reinterpret_cast<T&>(_storage) = value;
    _typeIndex = indexOf<T>();
    return *this;
  }

  static constexpr size_t storageSize() { return StorageSize; }

  void reset() { _typeIndex = -1; }

  template <typename T> bool is() const { return _typeIndex == indexOf<T>(); }

  template <typename T> bool isNot() const {
    return _typeIndex != indexOf<T>();
  }

  /// Check if this object is holding a meaningful type
  bool holdingValue() const { return _typeIndex >= 0; }

  template <typename T> void set() {
    static_assert(indexOf<T>() < sizeof...(Types), "Wrong type");
    _typeIndex = indexOf<T>();
  }

  template <typename T> T& get() {
    assert(indexOf<T>() == _typeIndex);
    return reinterpret_cast<T&>(_storage);
  }

  template <typename T> const T& get() const {
    assert(indexOf<T>() == _typeIndex);
    return reinterpret_cast<const T&>(_storage);
  }

  int index() const { return _typeIndex; }

  std::byte* raw() { return _storage; }
  const std::byte* raw() const { return _storage; }
};

} // namespace utils

#endif // UTILS_PODVARIANT_H
