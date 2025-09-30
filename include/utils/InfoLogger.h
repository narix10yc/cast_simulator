#ifndef CAST_UTILS_INFOLOGGER_H
#define CAST_UTILS_INFOLOGGER_H

#include <ostream>
#include <string>

namespace utils {

class InfoLogger {
  std::ostream& os_;

public:
  int verbose = 1;
  int depth = 1;
  static constexpr int INDENT_SPACES = 2;

  InfoLogger(std::ostream& os, int verbose = 1, int depth = 0)
      : os_(os), verbose(verbose), depth(depth) {}

  InfoLogger& put(const char* label) {
    os_ << std::string(depth * INDENT_SPACES, ' ') << " - " << label << "\n";
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

  // General types
  template <typename T>
  InfoLogger& put(const char* label, const T& value, int requireVerbose = 1) {
    if (verbose < requireVerbose)
      return *this;
    os_ << std::string(depth * INDENT_SPACES, ' ') << " - " << label << " : "
        << value << "\n";
    return *this;
  }

  // Specialization for bool to print True/False
  template <>
  InfoLogger&
  put<bool>(const char* label, const bool& value, int requireVerbose) {
    if (verbose < requireVerbose)
      return *this;
    os_ << std::string(depth * INDENT_SPACES, ' ') << " - " << label << " : "
        << (value ? "True" : "False") << "\n";
    return *this;
  }

  InfoLogger indent() const { return InfoLogger(os_, verbose, depth + 1); }

  InfoLogger indent(int verbose) const {
    return InfoLogger(os_, verbose, depth + 1);
  }
};
} // namespace utils

#endif // CAST_UTILS_INFOLOGGER_H
