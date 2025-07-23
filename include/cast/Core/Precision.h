#ifndef CAST_CORE_PRECISION_H
#define CAST_CORE_PRECISION_H

namespace cast {
enum class Precision : int {
  F32 = 32,
  Single = 32,
  Float = 32,

  F64 = 64,
  Double = 64,

  Unknown = -1
};
} // namespace cast

#include "utils/CSVParsable.h"

namespace utils {

template <> struct CSVField<cast::Precision> {
  static void parse(std::string_view token, cast::Precision& field) {
    if (token == "32")
      field = cast::Precision::F32;
    else if (token == "64")
      field = cast::Precision::F64;
    else
      field = cast::Precision::Unknown;
  }

  static void write(std::ostream& os, const cast::Precision& value) {
    os << static_cast<int>(value);
  }
};

} // namespace utils

#endif // CAST_CORE_PRECISION_H