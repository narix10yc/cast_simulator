#ifndef UTILS_MAYBE_ERROR_H
#define UTILS_MAYBE_ERROR_H

#include <cassert>
#include <memory> // for std::unique_ptr
#include <string>

namespace cast {

namespace impl {
class MaybeErrorStatus {
  uint8_t status;

public:
  constexpr MaybeErrorStatus(uint8_t status) : status(status) {}

  bool isErrorPresent() const { return (status >> 1) & 1; }
  bool isErrorChecked() const { return (status >> 0) & 1; }

  void setErrorPresent() { status |= 0b10; }
  void setErrorChecked() { status |= 0b01; }
};

static constexpr uint8_t ErrorAbsentNotChecked = 0b00;
static constexpr uint8_t ErrorAbsentChecked = 0b01;
static constexpr uint8_t ErrorPresentNotChecked = 0b10;
static constexpr uint8_t ErrorPresentChecked = 0b11;

struct ErrorCodeAndMsg {
  std::string msg;
  int code;
};

} // namespace impl

class Error {
public:
  std::unique_ptr<impl::ErrorCodeAndMsg> err;

  Error() : err(std::make_unique<impl::ErrorCodeAndMsg>("", 0)) {}

  Error(const std::string& msg, int code = -1)
      : err(std::make_unique<impl::ErrorCodeAndMsg>(msg, code)) {}
};

template <typename T> class [[nodiscard]] MaybeError {
  static_assert(!std::is_reference_v<T>,
                "MaybeError<T> does not support references");
  static constexpr bool NotVoid = !std::is_void_v<T>;
  struct Dummy {};
  using value_type = std::conditional_t<NotVoid, T, Dummy>;

  using error_type = Error;
  union {
    value_type value_;
    error_type err_;
  };
  mutable impl::MaybeErrorStatus status;

public:
  MaybeError()
    requires(!NotVoid)
      : status(impl::ErrorAbsentNotChecked) {}

  MaybeError(Error&& i)
      : err_(std::move(i)), status(impl::ErrorPresentNotChecked) {}

  MaybeError(const value_type& value)
    requires(NotVoid)
      : value_(value), status(impl::ErrorAbsentNotChecked) {}

  MaybeError(value_type&& value) noexcept
    requires(NotVoid)
      : value_(std::move(value)), status(impl::ErrorAbsentNotChecked) {}

  ~MaybeError() {
    assert(status.isErrorChecked() &&
           "MaybeError must be checked before destruction");
    if (status.isErrorPresent())
      err_.~error_type();
    else if constexpr (NotVoid)
      value_.~value_type();
  }

  // MaybeError cannot be copied, only moved.
  MaybeError(const MaybeError&) = delete;
  MaybeError& operator=(const MaybeError&) = delete;

  // Move constructor.
  MaybeError(MaybeError&& other) noexcept : status(other.status) {
    if (this == &other)
      return;
    if (other.status.isErrorPresent())
      new (&err_) error_type(std::move(other.err_));
    else if constexpr (NotVoid)
      new (&value_) value_type(std::move(other.value_));
    // set check the other status to indicate it has been moved.
    other.status.setErrorChecked();
  }

  // Move assignment operator.
  MaybeError& operator=(MaybeError&& other) noexcept {
    if (this == &other)
      return *this;
    if (status.isErrorPresent() == other.status.isErrorPresent()) {
      // If both are in the same state, we can just move the value.
      if (status.isErrorPresent())
        err_ = std::move(other.err_);
      else if constexpr (NotVoid)
        value_ = std::move(other.value_);
      status.setErrorChecked();
      return *this;
    }
    this->~MaybeError();
    new (this) MaybeError(std::move(other));
    return *this;
  }

  // Consume (ignore) the error. In non-debug builds this function will
  // not check if error is present. In debug builds this function asserts no
  // error is present.
  void consumeError() {
    assert(!status.isErrorPresent() &&
           "Consuming a MaybeError<T> when error is actually present");
    status.setErrorChecked();
  }

  bool hasError() const {
    status.setErrorChecked();
    return status.isErrorPresent();
  }

  bool hasValue() const { return !hasError(); }

  value_type takeValue()
    requires(NotVoid)
  {
    assert(hasValue() && "No value present in MaybeError");
    return std::move(value_);
  }

  const std::string& what() const {
    assert(status.isErrorChecked() &&
           "MaybeError is not checked when trying to get the error message");
    if (status.isErrorPresent())
      return err_.err->msg;
    static const std::string emptyString;
    return emptyString;
  }

  // Get the error code. Returns 0 when no error is present.
  int err_code() const {
    assert(status.isErrorChecked() &&
           "MaybeError is not checked when trying to get the error code");
    if (status.isErrorPresent())
      return err_.err->code;
    return 0;
  }

  explicit operator bool() const { return hasValue(); }
}; // class MaybeError

static inline Error makeError(const std::string& errorMsg, int errorCode = -1) {
  return Error(errorMsg, errorCode);
}

} // namespace cast

#endif // UTILS_MAYBE_ERROR_H