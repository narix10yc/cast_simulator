#ifndef CAST_ADT_PATTERN_MATRIX_H
#define CAST_ADT_PATTERN_MATRIX_H

#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

namespace cast {

class ComplexSquareMatrix;

/// A pattern matrix is a matrix that describes the pattern of entries in a
/// matrix. Its entries are `Zero`, `PlusOne`, `MinusOne`, or `Generic`.
///
/// `PatternMatrix` is a convenient object to use in kernel generation to avoid
/// duplication of kernels.
class PatternMatrix;
using PatternMatrixPtr = std::shared_ptr<PatternMatrix>;
using ConstPatternMatrixPtr = std::shared_ptr<const PatternMatrix>;

class PatternMatrix {
public:
  enum Pattern : uint8_t { Zero = 0, PlusOne = 1, MinusOne = 2, Generic = 3 };

private:
  size_t edgeSize_;
  // Packed 2-bit entries: 4 patterns per byte.
  std::vector<std::uint8_t> data_;

  static constexpr std::size_t kBitsPerEntry = 2;
  static constexpr std::size_t kEntriesPerByte = 8 / kBitsPerEntry;      // 4
  static constexpr std::uint8_t kEntryMask = (1u << kBitsPerEntry) - 1u; // 0b11

  static constexpr std::size_t bytesForEntries(std::size_t nEntries) {
    return (nEntries + kEntriesPerByte - 1) / kEntriesPerByte;
  }

  // only call it after this->edgeSize_ is set
  void allocData() {
    assert(edgeSize_ > 0);
    data_.assign(bytesForEntries(size()), 0);
  }

  static constexpr std::size_t byteIdx(std::size_t entryIdx) {
    return entryIdx / kEntriesPerByte;
  }
  static constexpr std::size_t bitShift(std::size_t entryIdx) {
    return (entryIdx % kEntriesPerByte) * kBitsPerEntry;
  }

public:
  PatternMatrix(size_t edgeSize) : edgeSize_(edgeSize) { allocData(); }

  static PatternMatrixPtr FromComplexSquareMatrix(
      const ComplexSquareMatrix& matrix, double zTol, double oTol);

  // equals to 2 * edgeSize() * edgeSize()
  size_t size() const { return 2ULL * edgeSize_ * edgeSize_; }

  size_t halfSize() const { return edgeSize_ * edgeSize_; }

  size_t edgeSize() const { return edgeSize_; }

  // Linear index helpers
  static constexpr std::size_t
  reIdx(std::size_t edgeSize, std::size_t row, std::size_t col) {
    return row * edgeSize + col;
  }

  static constexpr std::size_t
  imIdx(std::size_t edgeSize, std::size_t row, std::size_t col) {
    return edgeSize * edgeSize + row * edgeSize + col;
  }

  Pattern getEntry(std::size_t entryIdx) const {
    assert(entryIdx < size());
    const std::size_t b = byteIdx(entryIdx);
    const std::size_t sh = bitShift(entryIdx);
    const std::uint8_t v =
        static_cast<std::uint8_t>((data_[b] >> sh) & kEntryMask);
    return static_cast<Pattern>(v);
  }

  Pattern getRe(std::size_t row, std::size_t col) const {
    assert(row < edgeSize_ && col < edgeSize_);
    return getEntry(reIdx(edgeSize_, row, col));
  }

  Pattern getIm(std::size_t row, std::size_t col) const {
    assert(row < edgeSize_ && col < edgeSize_);
    return getEntry(imIdx(edgeSize_, row, col));
  }

  void setEntry(std::size_t entryIdx, Pattern p) {
    assert(entryIdx < size());
    const std::size_t b = byteIdx(entryIdx);
    const std::size_t sh = bitShift(entryIdx);
    const std::uint8_t pv = static_cast<std::uint8_t>(p) & kEntryMask;
    data_[b] = static_cast<std::uint8_t>((data_[b] & ~(kEntryMask << sh)) |
                                         (pv << sh));
  }

  void setRe(std::size_t row, std::size_t col, Pattern p) {
    assert(row < edgeSize_ && col < edgeSize_);
    setEntry(reIdx(edgeSize_, row, col), p);
  }

  void setIm(std::size_t row, std::size_t col, Pattern p) {
    assert(row < edgeSize_ && col < edgeSize_);
    setEntry(imIdx(edgeSize_, row, col), p);
  }

private:
  // Internal helper for iterator dereference.
  Pattern entryAtUnchecked(std::size_t entryIdx) const {
    const std::size_t b = byteIdx(entryIdx);
    const std::size_t sh = bitShift(entryIdx);
    const std::uint8_t v =
        static_cast<std::uint8_t>((data_[b] >> sh) & kEntryMask);
    return static_cast<Pattern>(v);
  }

public:
  class ConstIterator {
  public:
    using value_type = Pattern;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    ConstIterator(const PatternMatrix* owner, std::size_t idx)
        : owner_(owner), idx_(idx) {}

    Pattern operator*() const {
      assert(owner_ != nullptr);
      assert(idx_ < owner_->size());
      return owner_->entryAtUnchecked(idx_);
    }

    ConstIterator& operator++() {
      ++idx_;
      return *this;
    }

    bool operator==(const ConstIterator& other) const {
      return owner_ == other.owner_ && idx_ == other.idx_;
    }
    bool operator!=(const ConstIterator& other) const {
      return !(*this == other);
    }

  private:
    const PatternMatrix* owner_;
    std::size_t idx_;
  };

  // Iteration over the packed entries (returned by value).
  ConstIterator reBegin() const { return ConstIterator(this, 0); }
  ConstIterator reEnd() const { return ConstIterator(this, halfSize()); }

  ConstIterator imBegin() const { return ConstIterator(this, halfSize()); }
  ConstIterator imEnd() const { return ConstIterator(this, size()); }

  // Raw packed storage access (for hashing/caching/debugging).
  const std::vector<std::uint8_t>& packedBytes() const { return data_; }
};
} // namespace cast

#endif // CAST_ADT_PATTERN_MATRIX_H