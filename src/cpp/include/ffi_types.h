#ifndef CAST_SIMULATOR_SRC_CPP_INCLUDE_FFI_TYPES_H
#define CAST_SIMULATOR_SRC_CPP_INCLUDE_FFI_TYPES_H

// Shared C types exported across the Rust-C FFI boundary.
// Included by ffi_cpu.h and ffi_cuda.h.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cast_precision_t {
  CAST_PRECISION_F32 = 0,
  CAST_PRECISION_F64 = 1,
} cast_precision_t;

typedef struct cast_complex64_t {
  double re;
  double im;
} cast_complex64_t;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CAST_SIMULATOR_SRC_CPP_INCLUDE_FFI_TYPES_H
