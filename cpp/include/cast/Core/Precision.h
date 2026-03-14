#ifndef CAST_CORE_PRECISION_H
#define CAST_CORE_PRECISION_H

#include <ostream>

namespace cast {

enum class Precision : int {
  FP32 = 32,
  Single = 32,
  Float = 32,

  FP64 = 64,
  Double = 64,

  Unknown = -1
};

inline std::ostream& operator<<(std::ostream& os, Precision p) {
  switch (p) {
  case Precision::FP32:
    return os << "FP32";
  case Precision::FP64:
    return os << "FP64";
  default:
    return os << "Unknown";
  }
}

} // namespace cast

#include "utils/CSVParsable.h"

namespace utils {

template <> struct CSVField<cast::Precision> {
  static void parse(std::string_view token, cast::Precision& field) {
    if (token == "32")
      field = cast::Precision::FP32;
    else if (token == "64")
      field = cast::Precision::FP64;
    else
      field = cast::Precision::Unknown;
  }

  static void write(std::ostream& os, const cast::Precision& value) {
    os << static_cast<int>(value);
  }
};

} // namespace utils

#endif // CAST_CORE_PRECISION_H