#ifndef UTILS_STRINGREF_H
#define UTILS_STRINGREF_H

#include <cassert>
#include <iostream>
#include <string>

namespace utils {

/// We would like to make our utility methods standalone. This is a simplified
/// replication of \c llvm::StringRef
class StringRef {
  const char* _bufferBegin;
  size_t _length;

public:
  constexpr StringRef() : _bufferBegin(nullptr), _length(0) {}

  constexpr StringRef(const char* c)
      : _bufferBegin(c), _length(std::char_traits<char>::length(c)) {}

  constexpr StringRef(const char* c, size_t l) : _bufferBegin(c), _length(l) {
    assert(l <= std::char_traits<char>::length(c));
  }

  void increment() {
    assert(_length > 0);
    ++_bufferBegin;
    --_length;
  }

  constexpr size_t length() const { return _length; }

  constexpr bool empty() const { return _length == 0; }

  constexpr const char* begin() const { return _bufferBegin; }
  constexpr const char* end() const { return _bufferBegin + _length; }

  int compare(StringRef other) const;

  /// compare case-insensitive
  int compare_ci(StringRef other) const;

  operator std::string() const { return std::string(_bufferBegin, _length); }

  friend std::ostream& operator<<(std::ostream& os, StringRef str) {
    if (str.empty())
      return os;
    os.write(str.begin(), str._length);
    return os;
  }

  char operator[](size_t i) const {
    if (i == _length)
      return '\0';
    assert(i < _length);
    return _bufferBegin[i];
  }
}; // class StringRef

} // namespace utils

#endif // UTILS_STRINGREF_H
