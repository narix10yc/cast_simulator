#ifndef CAST_ADT_PATTERN_MATRIX_H
#define CAST_ADT_PATTERN_MATRIX_H

#include <cassert>
#include <memory>

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
  std::unique_ptr<Pattern[]> data_;

  // only call it after this->edgeSize_ is set
  void allocData() {
    assert(edgeSize_ > 0);
    data_ = std::make_unique<Pattern[]>(2ULL * edgeSize_ * edgeSize_);
    assert(data_ != nullptr && "Allocation failed");
  }

public:
  // Deliberated deleted. Use factory constructor `FromComplexSquareMatrix`
  PatternMatrix() = delete;

  static PatternMatrixPtr
  FromComplexSquareMatrix(const ComplexSquareMatrix& matrix);

  // equals to 2 * edgeSize() * edgeSize()
  size_t size() const { return 2ULL * edgeSize_ * edgeSize_; }

  size_t halfSize() const { return edgeSize_ * edgeSize_; }

  size_t edgeSize() const { return edgeSize_; }

  const Pattern* reBegin() const { return data_.get(); }
  const Pattern* reEnd() const { return data_.get() + halfSize(); }

  const Pattern* imBegin() const { return data_.get() + halfSize(); }
  const Pattern* imEnd() const { return data_.get() + size(); }
};
} // namespace cast

#endif // CAST_ADT_PATTERN_MATRIX_H