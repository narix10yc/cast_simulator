#ifndef CAST_SIMULATOR_SRC_CPP_INTERNAL_ERR_BUF_HPP
#define CAST_SIMULATOR_SRC_CPP_INTERNAL_ERR_BUF_HPP

// FFI error buffer helpers shared by the CPU and CUDA backends.
// All functions are inline to avoid a separate translation unit.

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>

inline void write_err_buf(char *err_buf, size_t err_buf_len, const std::string &msg) {
  if (err_buf == nullptr || err_buf_len == 0)
    return;
  const size_t n = std::min(err_buf_len - 1, msg.size());
  std::memcpy(err_buf, msg.data(), n);
  err_buf[n] = '\0';
}

inline void clear_err_buf(char *err_buf, size_t err_buf_len) {
  if (err_buf != nullptr && err_buf_len > 0)
    err_buf[0] = '\0';
}

#endif // CAST_SIMULATOR_SRC_CPP_INTERNAL_ERR_BUF_HPP
