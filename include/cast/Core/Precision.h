#ifndef CAST_CORE_PRECISION_H
#define CAST_CORE_PRECISION_H

namespace cast {
  enum class Precision : int {
    F32    = 32,
    Single = 32,
    Float  = 32,

    F64    = 64,
    Double = 64,

    Unknown = -1
  };
} // namespace cast

#endif // CAST_CORE_PRECISION_H