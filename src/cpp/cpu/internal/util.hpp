#ifndef CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_UTIL_HPP
#define CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_UTIL_HPP

// Internal helpers shared across cpu.cpp, cpu_gen.cpp, and cpu_jit.cpp.
// All functions are inline to avoid a separate translation unit.

#include "../../internal/err_buf.hpp"
#include "../../internal/types.hpp"

#include <cstddef>

namespace cast::cpu {

inline bool is_valid_simd_width(SimdWidth w) {
  return w == SimdWidth::W128 || w == SimdWidth::W256 || w == SimdWidth::W512;
}

inline bool is_valid_mode(MatrixLoadMode m) {
  return m == MatrixLoadMode::ImmValue || m == MatrixLoadMode::StackLoad;
}

/// Returns simd_s = log2(register_scalars) for a given SIMD width and
/// precision.  The result is always in [1, 4] for valid inputs; returns 0
/// for any invalid combination (should not occur after validation).
inline unsigned get_simd_s(SimdWidth w, cast::Precision p) {
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

/// Computes the expected flat matrix length (2^n_qubits)^2.
/// Returns false if n_qubits is too large to represent in a size_t.
inline bool expected_matrix_len(size_t n_qubits, size_t *out_len) {
  constexpr size_t kBits = sizeof(size_t) * 8;
  if (n_qubits >= kBits / 2) {
    return false;
  }
  *out_len = static_cast<size_t>(1) << (2 * n_qubits);
  return true;
}

} // namespace cast::cpu

#endif // CAST_SIMULATOR_SRC_CPP_CPU_INTERNAL_UTIL_HPP
