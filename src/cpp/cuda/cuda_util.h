#ifndef CAST_SIMULATOR_SRC_CPP_CUDA_UTIL_H
#define CAST_SIMULATOR_SRC_CPP_CUDA_UTIL_H

// Internal helpers shared across cuda.cpp, cuda_exec.cpp, etc.
// All functions are inline to avoid a separate translation unit.

#include <algorithm>
#include <cstring>
#include <string>

namespace cast_cuda_detail {

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

} // namespace cast_cuda_detail

#endif // CAST_SIMULATOR_SRC_CPP_CUDA_UTIL_H
