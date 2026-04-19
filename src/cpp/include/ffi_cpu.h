#ifndef CAST_SIMULATOR_SRC_CPP_INCLUDE_FFI_CPU_H
#define CAST_SIMULATOR_SRC_CPP_INCLUDE_FFI_CPU_H

// Rust-C FFI boundary for the CPU kernel pipeline.

#include "ffi_types.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t cast_cpu_kernel_id_t;

typedef enum cast_cpu_simd_width_t {
  CAST_CPU_SIMD_WIDTH_W128 = 128,
  CAST_CPU_SIMD_WIDTH_W256 = 256,
  CAST_CPU_SIMD_WIDTH_W512 = 512,
} cast_cpu_simd_width_t;

typedef enum cast_cpu_matrix_load_mode_t {
  CAST_CPU_MATRIX_LOAD_IMM_VALUE = 0,
  CAST_CPU_MATRIX_LOAD_STACK_LOAD = 1,
} cast_cpu_matrix_load_mode_t;

/// Per-kernel generation request.  Borrowed pointers (`qubits`, `matrix`)
/// must be valid for the duration of the call.  The Rust side owns the
/// canonical form and hashes it for dedup; this struct is purely on-the-wire.
typedef struct cast_cpu_kernel_gen_request_t {
  // Codegen spec
  cast_precision_t precision;
  cast_cpu_simd_width_t simd_width;
  cast_cpu_matrix_load_mode_t mode;
  double ztol;
  double otol;
  // Gate identity
  const uint32_t *qubits;
  size_t n_qubits;
  const cast_complex64_t *matrix;
  size_t matrix_len;
  // Diagnostics
  bool capture_ir;
  bool capture_asm;
} cast_cpu_kernel_gen_request_t;

typedef struct cast_cpu_kernel_metadata_t {
  cast_cpu_kernel_id_t kernel_id;
  cast_precision_t precision;
  cast_cpu_simd_width_t simd_width;
  cast_cpu_matrix_load_mode_t mode;
  uint32_t n_gate_qubits;
} cast_cpu_kernel_metadata_t;

/// Per-kernel data returned by cast_cpu_kernel_generator_finish.
/// The `matrix`, `ir_text`, and `asm_text` fields are heap-allocated
/// (malloc); call cast_cpu_jit_kernel_records_free to release them after
/// copying the data.
typedef struct cast_cpu_jit_kernel_record_t {
  cast_cpu_kernel_metadata_t metadata;
  void (*entry)(void *);    ///< JIT-compiled function pointer.
  cast_complex64_t *matrix; ///< NULL for ImmValue mode.
  size_t matrix_len;
  char *ir_text;  ///< NULL if capture_ir was false in the request.
  char *asm_text; ///< NULL if capture_asm was false in the request.
} cast_cpu_jit_kernel_record_t;

typedef struct cast_cpu_kernel_generator_t cast_cpu_kernel_generator_t;
typedef struct cast_cpu_jit_session_t cast_cpu_jit_session_t;

// -- Generator lifecycle --

/// Error handling: returns NULL on failure. Rust side must check if the
/// returned handle is null.
cast_cpu_kernel_generator_t *cast_cpu_kernel_generator_new(void);

/// Safe to call with NULL.
void cast_cpu_kernel_generator_delete(cast_cpu_kernel_generator_t *generator);

// -- Kernel generation --

/// Generates one kernel from `request`.  Returns the assigned kernel id
/// (> 0) on success, or 0 on failure with an error message in `err_buf`.
/// The request struct's pointer fields need only be valid for the
/// duration of this call.
cast_cpu_kernel_id_t
cast_cpu_kernel_generator_generate(cast_cpu_kernel_generator_t *generator,
                                   const cast_cpu_kernel_gen_request_t *request, char *err_buf,
                                   size_t err_buf_len);

// -- JIT compilation --

/// Error handling: returns non-zero on failure and writes a message to err_buf.
/// On success: deletes the generator and returns 0.
/// On failure: leaves the generator intact.
int cast_cpu_kernel_generator_finish(cast_cpu_kernel_generator_t *generator,
                                     cast_cpu_jit_session_t **out_session,
                                     cast_cpu_jit_kernel_record_t **out_records,
                                     size_t *out_n_records, char *err_buf, size_t err_buf_len);

/// Safe to call with NULL.
void cast_cpu_jit_kernel_records_free(cast_cpu_jit_kernel_record_t *records, size_t n);

// -- Session lifecycle --

/// Safe to call with NULL.
void cast_cpu_jit_session_delete(cast_cpu_jit_session_t *session);

// ── Internal to C++ (not imported by Rust) ──────────────────────────────────

typedef void cast_cpu_kernel_entry_t(void *);

typedef struct cast_cpu_launch_args_t {
  void *sv;
  uint64_t ctr_begin;
  uint64_t ctr_end;
  void *p_mat;
} cast_cpu_launch_args_t;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CAST_SIMULATOR_SRC_CPP_INCLUDE_FFI_CPU_H
