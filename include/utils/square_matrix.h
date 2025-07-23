#ifndef UTILS_SQUARE_MATRIX_H
#define UTILS_SQUARE_MATRIX_H

#include "utils/utils.h"
#include <cassert>
#include <complex>
#include <iostream>
#include <vector>

namespace utils {

/// @brief square_matrix<data_t>: a simple wrapper of square matrices.
template <typename data_t> class square_matrix {
private:
  size_t _edgeSize; // use size_t here to avoid overflow
  data_t* _data;

public:
  square_matrix() = default;

  square_matrix(unsigned edgeSize)
      : _edgeSize(edgeSize),
        _data(new data_t[static_cast<size_t>(edgeSize) * edgeSize]) {
    assert(_data != nullptr);
  }

  square_matrix(std::initializer_list<data_t> initData) {
    auto size = initData.size();
    if (size == 0) {
      _edgeSize = 0;
      _data = nullptr;
      return;
    }
    _edgeSize = std::sqrt(size);
    assert(_edgeSize * _edgeSize == size);
    _data = new data_t[_edgeSize * _edgeSize];
    size_t i = 0;
    for (const auto& item : initData)
      _data[i++] = item;
  }

  ~square_matrix() { delete[] _data; }

  square_matrix(const square_matrix& other)
      : _edgeSize(other.edgeSize()),
        _data(new data_t[other.edgeSize() * other.edgeSize()]) {
    for (size_t i = 0; i < _edgeSize * _edgeSize; i++)
      _data[i] = other[i];
  }

  square_matrix(square_matrix&& other) noexcept
      : _edgeSize(other.edgeSize()), _data(other._data) {
    other._data = nullptr;
  }

  square_matrix& operator=(const square_matrix& other) {
    if (this == &other)
      return *this;

    this->~square_matrix();
    new (this) square_matrix(other);
    return *this;
  }

  square_matrix& operator=(square_matrix&& other) noexcept {
    if (&other == this)
      return *this;

    this->~square_matrix();
    new (this) square_matrix(std::move(other));
    assert(other._data == nullptr);
    return *this;
  }

  data_t* data() { return _data; }
  const data_t* data() const { return _data; }

  // get the pointer to the row r
  data_t* row(unsigned r) {
    assert(r < _edgeSize);
    return _data + r * _edgeSize;
  }

  // get the pointer to the row r
  const data_t* row(unsigned r) const {
    assert(r < _edgeSize);
    return _data + r * _edgeSize;
  }

  // get element specified by row and column
  data_t& rc(unsigned r, unsigned c) {
    assert(r < _edgeSize && c < _edgeSize);
    return _data[r * _edgeSize + c];
  }

  // get element specified by row and column
  const data_t& rc(unsigned r, unsigned c) const {
    assert(r < _edgeSize && c < _edgeSize);
    return _data[r * _edgeSize + c];
  }

  // get element specified by row and column
  data_t& operator()(unsigned r, unsigned c) { return rc(r, c); }
  // get element specified by row and column
  const data_t& operator()(unsigned r, unsigned c) const { return rc(r, c); }

  data_t& operator[](size_t idx) { return _data[idx]; }
  const data_t& operator[](size_t idx) const { return _data[idx]; }

  data_t* begin() { return _data; }
  data_t* end() { return _data + size(); }
  const data_t* begin() const { return _data; }
  const data_t* end() const { return _data + size(); }
  const data_t* cbegin() const { return _data; }
  const data_t* cend() const { return _data + size(); }

  size_t size() const { return _edgeSize * _edgeSize; }

  size_t edgeSize() const { return _edgeSize; }

  size_t sizeInBytes() const { return _edgeSize * _edgeSize * sizeof(data_t); }

  square_matrix permute(const llvm::SmallVector<int>& flags) const {
    assert(utils::isPermutation(flags));
    assert(1 << flags.size() == _edgeSize);

    if (std::all_of(flags.begin(), flags.end(), [&flags](int i) {
          return flags[i] == i;
        }))
      return square_matrix(*this);

    const auto permuteIndex = [&flags,
                               k = flags.size()](unsigned idx) -> unsigned {
      unsigned newIdx = 0;
      for (unsigned b = 0; b < k; b++) {
        newIdx += ((idx & (1ULL << b)) >> b) << flags[b];
      }
      return newIdx;
    };

    square_matrix matrix(_edgeSize);
    for (unsigned r = 0; r < _edgeSize; r++) {
      for (unsigned c = 0; c < _edgeSize; c++) {
        matrix(permuteIndex(r), permuteIndex(c)) = this->rc(r, c);
      }
    }

    return matrix;
  }
};

/// The maximum norm (sometimes infinity norm) between two matrices is defined
/// by the largest difference in absolute values in matrix entries. That is,
/// max_{i,j} |A_ij - B_ij|
template <typename data_t>
double maximum_norm(const square_matrix<data_t>& m0,
                    const square_matrix<data_t>& m1) {
  assert(m0.edgeSize() == m1.edgeSize());
  double norm = 0.0;
  double tmp;
  for (unsigned r = 0; r < m0.edgeSize(); r++) {
    for (unsigned c = 0; c < m1.edgeSize(); c++) {
      tmp = std::abs(m0(r, c) - m1(r, c));
      if (tmp > norm)
        norm = tmp;
    }
  }
  return norm;
}

square_matrix<std::complex<double>> randomUnitaryMatrix(unsigned edgeSize);

std::ostream&
printComplexMatrixF64(std::ostream& os,
                      const square_matrix<std::complex<double>>& matrix);

inline std::ostream&
printComplexMatrixF64(const square_matrix<std::complex<double>>& matrix) {
  return printComplexMatrixF64(std::cerr, matrix);
}

} // namespace utils

#endif // UTILS_SQUARE_MATRIX_H