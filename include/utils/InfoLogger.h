#ifndef CAST_UTILS_INFOLOGGER_H
#define CAST_UTILS_INFOLOGGER_H

#include <concepts>
#include <functional>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

// To provide an specialization for cast::Precision
#include "cast/Core/Precision.h"

namespace utils {

class InfoLogger {
  std::ostream& os_;

  template <typename Label> static std::string label_to_string_(Label&& label) {
    if constexpr (std::is_same_v<std::decay_t<Label>, std::string>) {
      return label;
    } else if constexpr (std::is_convertible_v<Label, std::string_view>) {
      return std::string(std::string_view(label));
    } else {
      return std::string(std::forward<Label>(label));
    }
  }

  template <typename Label>
  static constexpr bool is_label_v =
      std::is_convertible_v<Label, std::string_view> ||
      std::constructible_from<std::string, Label>;

  template <typename T> void put_format_(std::ostream& os) {
    if constexpr (std::is_floating_point_v<T>) {
      os << std::defaultfloat;
    } else if constexpr (std::is_integral_v<T>) {
      os << std::dec;
    }
  }

public:
  int verbose = 1;
  int depth = 1;
  static constexpr int INDENT_SPACES = 2;

  InfoLogger(std::ostream& os, int verbose = 1, int depth = 0)
      : os_(os), verbose(verbose), depth(depth) {}

  // Get the raw ostream attached to this InfoLogger.
  std::ostream& raw() { return os_; }

  template <typename Label>
    requires is_label_v<Label>
  InfoLogger& put(Label&& label) {
    auto labelStr = label_to_string_(std::forward<Label>(label));
    os_ << std::string(depth * INDENT_SPACES, ' ') << " - " << labelStr << "\n";
    return *this;
  }

  // Specialization for bool to print True/False
  template <typename Label>
    requires is_label_v<Label>
  InfoLogger& put(Label&& label, bool value, int requireVerbose = 1) {
    if (verbose < requireVerbose)
      return *this;
    auto labelStr = label_to_string_(std::forward<Label>(label));
    os_ << std::string(depth * INDENT_SPACES, ' ') << " - " << labelStr << " : "
        << (value ? "True" : "False") << "\n";
    return *this;
  }

  // Specialization for cast::Precision
  template <typename Label>
    requires is_label_v<Label>
  InfoLogger& put(Label&& label, cast::Precision p, int requireVerbose = 1) {
    if (verbose < requireVerbose)
      return *this;
    auto labelStr = label_to_string_(std::forward<Label>(label));
    os_ << std::string(depth * INDENT_SPACES, ' ') << " - " << labelStr
        << " : ";
    switch (p) {
    case cast::Precision::FP32:
      os_ << "FP32";
      break;
    case cast::Precision::FP64:
      os_ << "FP64";
      break;
    default:
      os_ << "Unknown";
      break;
    }
    os_ << "\n";
    return *this;
  }

  // Custom printing function
  template <typename Label, typename Func>
    requires is_label_v<Label> && std::invocable<Func, std::ostream&>
  InfoLogger& put(Label&& label, Func&& func, int requireVerbose = 1) {
    if (verbose < requireVerbose)
      return *this;
    auto labelStr = label_to_string_(std::forward<Label>(label));
    os_ << std::string(depth * INDENT_SPACES, ' ') << " - " << labelStr
        << " : ";
    std::invoke(std::forward<Func>(func), os_);
    os_ << "\n";
    return *this;
  }

  // For span of elements
  template <typename Label, typename T>
    requires is_label_v<Label>
  InfoLogger&
  put(Label&& label, std::span<const T> span, int requireVerbose = 1) {
    if (verbose < requireVerbose)
      return *this;
    put_format_<T>(os_);
    auto labelStr = label_to_string_(std::forward<Label>(label));
    os_ << std::string(depth * INDENT_SPACES, ' ') << " - " << labelStr
        << " : ";
    os_ << "[";
    for (const auto& v : span)
      os_ << v << ", ";
    os_ << "]\n";
    return *this;
  }

  // General types: call operator<<(std::ostream&, const T&)
  template <typename Label, typename T>
    requires is_label_v<Label> &&
             requires(std::ostream& os, const T& value) { os << value; }
  InfoLogger& put(Label&& label, const T& value, int requireVerbose = 1) {
    if (verbose < requireVerbose)
      return *this;
    put_format_<T>(os_);
    auto labelStr = label_to_string_(std::forward<Label>(label));
    os_ << std::string(depth * INDENT_SPACES, ' ') << " - " << labelStr << " : "
        << value << "\n";
    return *this;
  }

  /// Log with indentation, using the same verbosity level
  template <typename Func>
    requires std::invocable<Func&, InfoLogger&>
  InfoLogger& indent(Func&& f) {
    InfoLogger logger(os_, this->verbose, depth + 1);
    std::invoke(std::forward<Func>(f), logger);
    return *this;
  }

  /// Log with indentation and a different verbosity level
  template <typename Func>
    requires std::invocable<Func&, InfoLogger&>
  InfoLogger& indent(int verboseLevel, Func&& f) {
    InfoLogger logger(os_, verboseLevel, depth + 1);
    std::invoke(std::forward<Func>(f), logger);
    return *this;
  }

  /// Log with custom indentation depth delta.
  template <typename Func>
    requires std::invocable<Func&, InfoLogger&>
  InfoLogger& indentBy(int depthDelta, Func&& f) {
    InfoLogger logger(os_, this->verbose, depth + depthDelta);
    std::invoke(std::forward<Func>(f), logger);
    return *this;
  }
};
} // namespace utils

#endif // CAST_UTILS_INFOLOGGER_H
