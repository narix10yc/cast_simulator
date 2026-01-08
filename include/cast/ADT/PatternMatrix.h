#ifndef CAST_ADT_PATTERN_MATRIX_H
#define CAST_ADT_PATTERN_MATRIX_H

#include <cassert>
#include <compare>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>

namespace cast {

class ComplexSquareMatrix;

/// A pattern matrix is a matrix that describes the pattern of entries in a
/// complex matrix. Its entries are `Zero`, `PlusOne`, `MinusOne`, or `Generic`.
///
/// `PatternMatrix` is a convenient object used in kernel generation to avoid
/// duplication of kernels. It is quickly hashable and comparable.
/// TODO: a cached pool for constant pattern matrices
class PatternMatrix;
using PatternMatrixPtr = std::shared_ptr<PatternMatrix>;
using ConstPatternMatrixPtr = std::shared_ptr<const PatternMatrix>;

class PatternMatrix {
public:
  enum Pattern : uint8_t { Zero = 0, PlusOne = 1, MinusOne = 2, Generic = 3 };

private:
  size_t edgeSize_;
  // For certain edgeSize_, we need to pad the byte array to hold all entries.
  // For example, edgeSize_ = 3 => 18 entries => 5 x uint8_t bytes (holds 20
  // patterns).
  size_t byteCount_;
  // Packed 2-bit entries: 4 patterns per byte.
  std::unique_ptr<std::uint8_t[]> data_;

  // The constructor, for internal use only. This function does *not*
  // zero-initialize the allocated data.
  void construct(size_t edgeSize) {
    edgeSize_ = edgeSize;
    size_t nEntries = 2ULL * edgeSize_ * edgeSize_;
    byteCount_ = bytesForEntries(nEntries);
    data_ = std::make_unique<std::uint8_t[]>(byteCount_);
  }

  static constexpr size_t kBitsPerEntry = 2;
  static constexpr size_t kEntriesPerByte = 8 / kBitsPerEntry;           // 4
  static constexpr std::uint8_t kEntryMask = (1u << kBitsPerEntry) - 1u; // 0b11

  static constexpr size_t bytesForEntries(size_t nEntries) {
    return (nEntries + kEntriesPerByte - 1) / kEntriesPerByte;
  }

  static constexpr size_t byteIdx(size_t entryIdx) {
    return entryIdx / kEntriesPerByte;
  }
  static constexpr size_t bitShift(size_t entryIdx) {
    return (entryIdx % kEntriesPerByte) * kBitsPerEntry;
  }

  size_t reIdx(size_t r, size_t c) const { return r * edgeSize_ + c; }

  size_t imIdx(size_t r, size_t c) const {
    return edgeSize_ * edgeSize_ + r * edgeSize_ + c;
  }

public:
  PatternMatrix() : edgeSize_(0), byteCount_(0), data_(nullptr) {}

  // Construct and zero-initialize
  PatternMatrix(size_t edgeSize) {
    construct(edgeSize);
    // zero-initialize
    std::memset(data_.get(), 0, byteCount_);
  }

  PatternMatrix(const PatternMatrix& other) {
    construct(other.edgeSize_);
    std::memcpy(data_.get(), other.data_.get(), byteCount_);
  }

  PatternMatrix(PatternMatrix&& other) noexcept = default;

  PatternMatrix& operator=(const PatternMatrix& other) {
    if (this == &other)
      return *this;
    construct(other.edgeSize_);
    std::memcpy(data_.get(), other.data_.get(), byteCount_);
    return *this;
  }

  PatternMatrix& operator=(PatternMatrix&& other) noexcept = default;

  ~PatternMatrix() = default;

  static PatternMatrixPtr FromComplexSquareMatrix(
      const ComplexSquareMatrix& matrix, double zTol, double oTol);

  // Size of both real and imag parts. Equals to 2 * edgeSize() * edgeSize().
  // For the size of real or imag parts each, use halfSize().
  size_t size() const { return 2ULL * edgeSize_ * edgeSize_; }

  // Get the size of the real (or imag) part alone.
  // Equals to edgeSize() * edgeSize()
  size_t halfSize() const { return edgeSize_ * edgeSize_; }

  size_t edgeSize() const { return edgeSize_; }

  Pattern getEntry(size_t entryIdx) const {
    assert(entryIdx < size());
    const size_t b = byteIdx(entryIdx);
    const size_t sh = bitShift(entryIdx);
    const std::uint8_t v =
        static_cast<std::uint8_t>((data_[b] >> sh) & kEntryMask);
    return static_cast<Pattern>(v);
  }

  Pattern getRe(size_t r, size_t c) const {
    assert(r < edgeSize_ && c < edgeSize_);
    return getEntry(reIdx(r, c));
  }

  Pattern getIm(size_t r, size_t c) const {
    assert(r < edgeSize_ && c < edgeSize_);
    return getEntry(imIdx(r, c));
  }

  Pattern real(size_t r, size_t c) const { return getRe(r, c); }

  Pattern imag(size_t r, size_t c) const { return getIm(r, c); }

  void setEntry(size_t entryIdx, Pattern p) {
    assert(entryIdx < size());
    const size_t b = byteIdx(entryIdx);
    const size_t sh = bitShift(entryIdx);
    const std::uint8_t pv = static_cast<std::uint8_t>(p) & kEntryMask;
    data_[b] = static_cast<std::uint8_t>((data_[b] & ~(kEntryMask << sh)) |
                                         (pv << sh));
  }

  void setRe(size_t r, size_t c, Pattern p) {
    assert(r < edgeSize_ && c < edgeSize_);
    setEntry(reIdx(r, c), p);
  }

  void setIm(size_t r, size_t c, Pattern p) {
    assert(r < edgeSize_ && c < edgeSize_);
    setEntry(imIdx(r, c), p);
  }

  void setReal(size_t r, size_t c, Pattern p) { setRe(r, c, p); }

  void setImag(size_t r, size_t c, Pattern p) { setIm(r, c, p); }

private:
  // Internal helper for iterator dereference.
  Pattern entryAtUnchecked(size_t entryIdx) const {
    const size_t b = byteIdx(entryIdx);
    const size_t sh = bitShift(entryIdx);
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

    ConstIterator(const PatternMatrix* owner, size_t idx)
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
    size_t idx_;
  };

  // Iteration over the packed entries (returned by value).
  ConstIterator reBegin() const { return ConstIterator(this, 0); }
  ConstIterator reEnd() const { return ConstIterator(this, halfSize()); }

  ConstIterator imBegin() const { return ConstIterator(this, halfSize()); }
  ConstIterator imEnd() const { return ConstIterator(this, size()); }

  ConstIterator realBegin() const { return reBegin(); }
  ConstIterator realEnd() const { return reEnd(); }

  ConstIterator imagBegin() const { return imBegin(); }
  ConstIterator imagEnd() const { return imEnd(); }

public:
  /* Comparators (Ordering) */
  bool operator==(const PatternMatrix& other) const {
    if (edgeSize_ != other.edgeSize_ || byteCount_ != other.byteCount_)
      return false;
    if (byteCount_ == 0)
      return true;
    return std::memcmp(data_.get(), other.data_.get(), byteCount_) == 0;
  }

  std::strong_ordering operator<=>(const PatternMatrix& other) const {
    if (auto cmp = edgeSize_ <=> other.edgeSize_; cmp != 0)
      return cmp;
    if (byteCount_ != other.byteCount_)
      return byteCount_ <=> other.byteCount_;
    if (byteCount_ == 0)
      return std::strong_ordering::equal;
    int cmp = std::memcmp(data_.get(), other.data_.get(), byteCount_);
    if (cmp < 0)
      return std::strong_ordering::less;
    if (cmp > 0)
      return std::strong_ordering::greater;
    return std::strong_ordering::equal;
  }

  // Raw packed storage access (for hashing/caching/debugging).
  const std::uint8_t* packedBytes() const { return data_.get(); }
  size_t packedByteCount() const { return byteCount_; }
};
} // namespace cast

#endif // CAST_ADT_PATTERN_MATRIX_H
