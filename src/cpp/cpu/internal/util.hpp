#ifndef CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_UTIL_HPP
#define CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_UTIL_HPP

// Internal helpers shared across cpu.cpp, cpu_gen.cpp, and cpu_jit.cpp.
// All functions are inline to avoid a separate translation unit.

#include "../../internal/err_buf.hpp"
#include "../../internal/types.hpp"

#include <cstddef>

namespace cast::cpu {

inline bool isValidSimdWidth(SimdWidth w) {
  return w == SimdWidth::W128 || w == SimdWidth::W256 || w == SimdWidth::W512;
}

inline bool isValidMode(MatrixLoadMode m) {
  return m == MatrixLoadMode::ImmValue || m == MatrixLoadMode::StackLoad;
}

/// Returns simdS = log2(register_scalars) for a given SIMD width and
/// precision.  The result is always in [1, 4] for valid inputs; returns 0
/// for any invalid combination (should not occur after validation).
inline unsigned getSimdS(SimdWidth w, cast::Precision p) {
  if (p == cast::Precision::F32) {
    switch (w) {
    case SimdWidth::W128:
      return 2;
    case SimdWidth::W256:
      return 3;
    case SimdWidth::W512:
      return 4;
    default:
      break;
    }
  }
  switch (w) {
  case SimdWidth::W128:
    return 1;
  case SimdWidth::W256:
    return 2;
  case SimdWidth::W512:
    return 3;
  default:
    break;
  }
  return 0;
}

/// Computes the expected flat matrix length (2^nQubits)^2.
/// Returns false if nQubits is too large to represent in a size_t.
inline bool expectedMatrixLen(size_t nQubits, size_t *outLen) {
  constexpr size_t kBits = sizeof(size_t) * 8;
  if (nQubits >= kBits / 2) {
    return false;
  }
  *outLen = static_cast<size_t>(1) << (2 * nQubits);
  return true;
}

} // namespace cast::cpu

#endif // CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_UTIL_HPP
