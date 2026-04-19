#ifndef CAST_SIMULATOR_SRC_CPP_INTERNAL_ERR_BUF_HPP
#define CAST_SIMULATOR_SRC_CPP_INTERNAL_ERR_BUF_HPP

// FFI error buffer helpers shared by the CPU and CUDA backends.
// All functions are inline to avoid a separate translation unit.

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>

inline void writeErrBuf(char *errBuf, size_t errBufLen, const std::string &msg) {
  if (errBuf == nullptr || errBufLen == 0)
    return;
  const size_t n = std::min(errBufLen - 1, msg.size());
  std::memcpy(errBuf, msg.data(), n);
  errBuf[n] = '\0';
}

inline void clearErrBuf(char *errBuf, size_t errBufLen) {
  if (errBuf != nullptr && errBufLen > 0)
    errBuf[0] = '\0';
}

#endif // CAST_SIMULATOR_SRC_CPP_INTERNAL_ERR_BUF_HPP
