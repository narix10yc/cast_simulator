#ifndef CAST_UTILS_INFOLOGGER_H
#define CAST_UTILS_INFOLOGGER_H

#include <functional>
#include <iomanip>
#include <ostream>
#include <span>
#include <string>

// To provide an specialization for cast::Precision
#include "cast/Core/Precision.h"

namespace utils {

class InfoLogger {
  std::ostream& os_;

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

  InfoLogger& put(const char* label) {
    os_ << std::string(depth * INDENT_SPACES, ' ') << " - " << label << "\n";
    return *this;
  }

  // Specialization for bool to print True/False
  InfoLogger& put(const char* label, bool value, int requireVerbose = 1) {
    if (verbose < requireVerbose)
      return *this;
    os_ << std::string(depth * INDENT_SPACES, ' ') << " - " << label << " : "
        << (value ? "True" : "False") << "\n";
    return *this;
  }

  // Specialization for cast::Precision
  InfoLogger&
  put(const char* label, cast::Precision p, int requireVerbose = 1) {
    if (verbose < requireVerbose)
      return *this;
    os_ << std::string(depth * INDENT_SPACES, ' ') << " - " << label << " : ";
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
  template <typename Func>
    requires std::invocable<Func, std::ostream&>
  InfoLogger& put(const char* label, Func&& func, int requireVerbose = 1) {
    if (verbose < requireVerbose)
      return *this;
    os_ << std::string(depth * INDENT_SPACES, ' ') << " - " << label << " : ";
    std::invoke(std::forward<Func>(func), os_);
    os_ << "\n";
    return *this;
  }

  // For span of elements
  template <typename T>
  InfoLogger&
  put(const char* label, std::span<const T> span, int requireVerbose = 1) {
    if (verbose < requireVerbose)
      return *this;
    put_format_<T>(os_);
    os_ << std::string(depth * INDENT_SPACES, ' ') << " - " << label << " : ";
    os_ << "[";
    for (const auto& v : span)
      os_ << v << ", ";
    os_ << "]\n";
    return *this;
  }

  // General types: call operator<<(std::ostream&, const T&)
  template <typename T>
  InfoLogger& put(const char* label, const T& value, int requireVerbose = 1) {
    if (verbose < requireVerbose)
      return *this;
    put_format_<T>(os_);
    os_ << std::string(depth * INDENT_SPACES, ' ') << " - " << label << " : "
        << value << "\n";
    return *this;
  }

  /// Log with indentation, using the same verbosity level
  InfoLogger& indent(std::function<void(InfoLogger&)> f) {
    InfoLogger logger(os_, this->verbose, depth + 1);
    f(logger);
    return *this;
  }

  /// Log with indentation and a different verbosity level
  InfoLogger& indent(int verbose, std::function<void(InfoLogger&)> f) {
    InfoLogger logger(os_, verbose, depth + 1);
    f(logger);
    return *this;
  }
};
} // namespace utils

#endif // CAST_UTILS_INFOLOGGER_H
