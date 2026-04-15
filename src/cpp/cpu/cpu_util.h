#ifndef CAST_SIMULATOR_SRC_CPP_CPU_UTIL_H
#define CAST_SIMULATOR_SRC_CPP_CPU_UTIL_H

// Internal helpers shared across cpu.cpp, cpu_gen.cpp, and cpu_jit.cpp.
// All functions are inline to avoid a separate translation unit.

#include "cast_cpu.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>

namespace cast_cpu_detail {

inline void write_error_message(char *err_buf, size_t err_buf_len, const std::string &msg) {
  if (err_buf == nullptr || err_buf_len == 0)
    return;
  const size_t n = std::min(err_buf_len - 1, msg.size());
  std::memcpy(err_buf, msg.data(), n);
  err_buf[n] = '\0';
}

inline void clear_error_buffer(char *err_buf, size_t err_buf_len) {
  if (err_buf != nullptr && err_buf_len > 0)
    err_buf[0] = '\0';
}

inline bool is_valid_precision(cast_cpu_precision_t p) {
  return p == CAST_CPU_PRECISION_F32 || p == CAST_CPU_PRECISION_F64;
}

inline bool is_valid_simd_width(cast_cpu_simd_width_t w) {
  return w == CAST_CPU_SIMD_WIDTH_W128 || w == CAST_CPU_SIMD_WIDTH_W256 ||
         w == CAST_CPU_SIMD_WIDTH_W512;
}

inline bool is_valid_mode(cast_cpu_matrix_load_mode_t m) {
  return m == CAST_CPU_MATRIX_LOAD_IMM_VALUE || m == CAST_CPU_MATRIX_LOAD_STACK_LOAD;
}

/// Returns simd_s = log2(register_scalars) for a given SIMD width and
/// precision.  The result is always in [1, 4] for valid inputs; returns 0
/// for any invalid combination (should not occur after validation).
inline unsigned get_simd_s(cast_cpu_simd_width_t w, cast_cpu_precision_t p) {
  if (p == CAST_CPU_PRECISION_F32) {
    switch (w) {
    case CAST_CPU_SIMD_WIDTH_W128:
      return 2;
    case CAST_CPU_SIMD_WIDTH_W256:
      return 3;
    case CAST_CPU_SIMD_WIDTH_W512:
      return 4;
    default:
      break;
    }
  }
  switch (w) {
  case CAST_CPU_SIMD_WIDTH_W128:
    return 1;
  case CAST_CPU_SIMD_WIDTH_W256:
    return 2;
  case CAST_CPU_SIMD_WIDTH_W512:
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
  if (n_qubits > kBits / 2) {
    return false;
  }
  *out_len = static_cast<size_t>(1) << (2 * n_qubits);
  return true;
}

} // namespace cast_cpu_detail

#endif // CAST_SIMULATOR_SRC_CPP_CPU_UTIL_H
