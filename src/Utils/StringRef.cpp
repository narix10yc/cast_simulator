#include "utils/StringRef.h"
using namespace utils;

int StringRef::compare(StringRef other) const {
  if (_length > other._length)
    return 1;
  if (_length < other._length)
    return -1;
  return std::char_traits<char>::compare(
      _bufferBegin, other._bufferBegin, _length);
}

int StringRef::compare_ci(StringRef other) const {
  if (_length > other._length)
    return 1;
  if (_length < other._length)
    return -1;
  auto* p0 = _bufferBegin;
  auto* p1 = other._bufferBegin;
  while (p0 < this->end()) {
    auto c0 = *p0;
    auto c1 = *p1;
    if (c0 >= 'a' && c0 <= 'z')
      c0 += 'A' - 'a';
    if (c1 >= 'a' && c1 <= 'z')
      c1 += 'A' - 'a';
    if (c0 > c1)
      return 1;
    if (c0 < c1)
      return -1;
    ++p0;
    ++p1;
  }
  return 0;
}