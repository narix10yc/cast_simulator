#ifndef UTILS_MAYBE_ERROR_H
#define UTILS_MAYBE_ERROR_H

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

template <typename T> class MaybeErrorInitializer {
public:
  std::string errMsg;
  MaybeErrorInitializer(const std::string& msg) : errMsg(msg) {}
};
} // namespace impl

template <typename T> class [[nodiscard]] MaybeError {
  static constexpr bool NotVoid = !std::is_void_v<T>;
  struct Dummy {};
  // In debug mode, MaybeError<T> is allowed to be destroyed only when
  // _errorMsg is a std::nullopt.
  using value_type = std::conditional_t<
      NotVoid,
      std::conditional_t<std::is_reference_v<T>,
                         std::reference_wrapper<std::remove_reference_t<T>>,
                         T>,
      Dummy>;
  using error_msg_type = std::string;
  union {
    value_type value_;
    error_msg_type errMsg_;
  };
  mutable impl::MaybeErrorStatus status;

public:
  MaybeError()
    requires(!NotVoid)
      : status(impl::ErrorAbsentNotChecked) {}

  MaybeError(const impl::MaybeErrorInitializer<T>& i)
      : status(impl::ErrorPresentNotChecked) {
    new (&errMsg_) error_msg_type(std::move(i.errMsg));
  }

  MaybeError(const value_type& value)
    requires(NotVoid)
      : status(impl::ErrorAbsentNotChecked) {
    new (&value_) value_type(value);
  }

  MaybeError(value_type&& value) noexcept
    requires(NotVoid)
      : status(impl::ErrorAbsentNotChecked) {
    new (&value_) value_type(std::move(value));
  }

  ~MaybeError() {
    assert(status.isErrorChecked() &&
           "MaybeError must be checked before destruction");
    if (status.isErrorPresent())
      errMsg_.~error_msg_type();
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
      new (&errMsg_) error_msg_type(std::move(other.errMsg_));
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
        errMsg_ = std::move(other.errMsg_);
      else if constexpr (NotVoid)
        value_ = std::move(other.value_);
      status.setErrorChecked();
      return *this;
    }
    this->~MaybeError();
    new (this) MaybeError(std::move(other));
    return *this;
  }

  // Consume (ignore) the error message. This function will not check if error
  // is present.
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

  value_type&& takeValue()
    requires(NotVoid)
  {
    assert(hasValue() && "No value present in MaybeError");
    return std::move(value_);
  }

  std::string takeError() {
    assert(hasError() && "No error to take");
    status.setErrorChecked();
    return std::move(errMsg_);
  }

  operator bool() const { return hasValue(); }
}; // class MaybeError

// Because we disallow MaybeError to be copied, we need a helper class to
// initialize with an error message.
template <typename T = void>
impl::MaybeErrorInitializer<T> makeError(const std::string& errorMsg) {
  return impl::MaybeErrorInitializer<T>(errorMsg);
}

} // namespace cast

#endif // UTILS_MAYBE_ERROR_H