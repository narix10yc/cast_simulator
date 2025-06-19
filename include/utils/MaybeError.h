#ifndef UTILS_MAYBE_ERROR_H
#define UTILS_MAYBE_ERROR_H

#include <optional>
#include <string>
#include <iostream>

namespace cast {

  namespace impl {
    class MaybeErrorStatus {
      uint8_t status;
    public:
      constexpr MaybeErrorStatus(uint8_t status) : status(status) {}

      bool isErrorPresent() const { return (status & 0b10) != 0; }
      bool isErrorChecked() const { return (status & 0b01) != 0; }

      void setErrorPresent() { status |= 0b10; }
      void setErrorChecked() { status |= 0b01; }
    };

    static constexpr uint8_t ErrorAbsentNotChecked = 0b00;  
    static constexpr uint8_t ErrorAbsentChecked = 0b01;
    static constexpr uint8_t ErrorPresentNotChecked = 0b10;
    static constexpr uint8_t ErrorPresentChecked = 0b11;

    template<typename T>
    class MaybeErrorInitializer {
    public:
      std::string errorMsg;
      MaybeErrorInitializer(const std::string& msg) : errorMsg(msg) {}
    };
  } // namespace impl

template<typename T>
class [[nodiscard]] MaybeError {
  // In debug mode, MaybeError<T> is allowed to be destroyed only when
  // _errorMsg is a std::nullopt.
  using value_type = std::conditional_t<
    std::is_reference_v<T>,
    std::reference_wrapper<std::remove_reference_t<T>>,
    T
  >;
  using error_msg_type = std::string;
  union {
    value_type _value;
    error_msg_type _errorMsg;
  };
  mutable impl::MaybeErrorStatus status;
public:
  MaybeError(const impl::MaybeErrorInitializer<T>& i)
    : status(impl::ErrorPresentNotChecked) {
    new (&_errorMsg) error_msg_type(std::move(i.errorMsg));
  }

  MaybeError(const T& value) : status(impl::ErrorAbsentNotChecked) {
    new (&_value) value_type(value);
  }

  MaybeError(T&& value) : status(impl::ErrorAbsentNotChecked) {
    new (&_value) value_type(std::move(value));
  }

  ~MaybeError() {
    assert(status.isErrorChecked() &&
           "MaybeError must be checked before destruction");
    if (status.isErrorPresent())
      _errorMsg.~error_msg_type();
    else
      _value.~value_type();
  }
  
  // MaybeError cannot be copied, only moved.
  MaybeError(const MaybeError&) = delete;
  MaybeError& operator=(const MaybeError&) = delete;

  // Move constructor.
  MaybeError(MaybeError&& other) noexcept : status(other.status) {
    if (this == &other)
      return;
    if (other.status.isErrorPresent())
      new (&_errorMsg) error_msg_type(std::move(other._errorMsg));
    else
      new (&_value) value_type(std::move(other._value)); 
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
        _errorMsg = std::move(other._errorMsg);
      else
        _value = std::move(other._value);
      status.setErrorChecked();
      return *this;
    }
    // If the states are different, we need to destroy the current value and
    this->~MaybeError();
    new (this) MaybeError(std::move(other));
    return *this;
  }

  // Consume (ignore) the error message. This function is only to be 
  // called when error is present.
  void consumeError() {
    assert(status.isErrorPresent() && "No error to consume");
    status.setErrorChecked();
  }

  bool hasError() const { 
    status.setErrorChecked();  
    return status.isErrorPresent();
  }

  bool hasValue() const { return !hasError(); }

  const T& getValue() const {
    assert(hasValue() && "No value present in MaybeError");
    return _value;
  }

  std::string takeError() {
    assert(hasError() && "No error to take");
    status.setErrorChecked();
    return std::move(_errorMsg);
  }

  const T& operator*() const { return getValue(); }

  operator bool() const { return hasValue(); }
}; // class MaybeError

// Because we disallow MaybeError to be copied, we need a helper class to 
// initialize with an error message.
template<typename T>
impl::MaybeErrorInitializer<T> makeError(const std::string& errorMsg) {
  return impl::MaybeErrorInitializer<T>(errorMsg);
}

} // namespace cast

#endif // UTILS_MAYBE_ERROR_H