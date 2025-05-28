#ifndef CAST_UNITARYPERMMATRIX_H
#define CAST_UNITARYPERMMATRIX_H

#include "utils/utils.h"

#include <vector>

namespace cast::legacy {

/// @brief A square matrix that each row or column has exactly one non-zero
/// entry of the form expi(phi).
/// @param data The (i, data[i].first) entry of the matrix is
/// expi(data[i].second)
class UnitaryPermutationMatrix {
public:
  struct Entry {
    size_t index;
    double phase;

    double normedPhase() const {
      auto p = std::fmod(phase, M_2_PI);
      if (p >= M_PI)
        return p - M_2_PI;
      if (p < -M_PI)
        return p + M_2_PI;
      return p;
    }
  };
private:
  Entry* _data;
  size_t _edgeSize;

public:
  ~UnitaryPermutationMatrix() { std::free(_data); }

  UnitaryPermutationMatrix() : _data(nullptr), _edgeSize(0) {}
  UnitaryPermutationMatrix(size_t edgeSize)
    : _data(static_cast<Entry*>(std::malloc(edgeSize * sizeof(Entry))))
    , _edgeSize(edgeSize) {}

  UnitaryPermutationMatrix(std::initializer_list<Entry> data) {
    _edgeSize = data.size();
    if (_edgeSize == 0) {
      _data = nullptr;
      return;
    }
    _data = static_cast<Entry*>(std::malloc(_edgeSize * sizeof(Entry)));
    size_t i = 0;
    for (const auto& item : data)
      _data[i++] = item;
  }

  UnitaryPermutationMatrix(const UnitaryPermutationMatrix& other) {
    _data = static_cast<Entry*>(std::malloc(other._edgeSize * sizeof(Entry)));
    _edgeSize = other._edgeSize;
    std::memcpy(_data, other._data, other._edgeSize * sizeof(Entry));
  }

  UnitaryPermutationMatrix& operator=(const UnitaryPermutationMatrix& other) {
    if (this == &other)
      return *this;

    this->~UnitaryPermutationMatrix();
    new (this) UnitaryPermutationMatrix(other);
    return *this;
  }

  UnitaryPermutationMatrix(UnitaryPermutationMatrix&& other) noexcept {
    _data = other._data;
    _edgeSize = other._edgeSize;
    other._data = nullptr;
  }

  UnitaryPermutationMatrix& operator=(
      UnitaryPermutationMatrix&& other) noexcept {
    if (this == &other)
      return *this;

    this->~UnitaryPermutationMatrix();
    new (this) UnitaryPermutationMatrix(std::move(other));
    assert(other._data == nullptr);
    return *this;
  }

  size_t edgeSize() const { return _edgeSize; }

  Entry& operator[](size_t index) {
    assert(index < _edgeSize);
    return _data[index];
  }
  const Entry& operator[](size_t index) const {
    assert(index < _edgeSize);
    return _data[index];
  }

  Entry* begin() { return _data; }
  Entry* end() { return _data + _edgeSize; }

  const Entry* begin() const { return _data; }
  const Entry* end() const { return _data + _edgeSize; }

  UnitaryPermutationMatrix permute(const llvm::SmallVector<int>& flags) const {
    assert(flags.size() == _edgeSize);
    if (_edgeSize == 0)
      return UnitaryPermutationMatrix();
    if (std::ranges::all_of(flags, [&flags](int i) { return flags[i] == i; }))
      return UnitaryPermutationMatrix(*this);

    const auto permuteIndex = [&flags, k = flags.size()](size_t src) -> size_t {
      size_t dst = 0;
      for (unsigned b = 0; b < k; b++)
        dst += ((src & (1ULL << b)) >> b) << flags[b];
      return dst;
    };
    const auto s = edgeSize();
    UnitaryPermutationMatrix matrix(s);

    for (size_t r = 0; r < s; r++)
      matrix[permuteIndex(r)] = {permuteIndex(_data[r].index), _data[r].phase};

    return matrix;
  }
};

} // namespace cast

#endif // CAST_UNITARYPERMMATRIX_H