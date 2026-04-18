#ifndef CAST_SIMULATOR_SRC_CPP_CPU_H
#define CAST_SIMULATOR_SRC_CPP_CPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Exported to Rust ────────────────────────────────────────────────────────
//
// Every type and function below has a matching Rust declaration:
//   - Types     → #[repr(C)] structs/enums in src/cpu/kernel.rs or src/cpu/types.rs
//   - Opaques   → zero-sized #[repr(C)] structs in kernel.rs::mod ffi
//   - Functions → unsafe extern "C" in kernel.rs::mod ffi

typedef uint64_t cast_cpu_kernel_id_t;

typedef enum cast_cpu_precision_t {
  CAST_CPU_PRECISION_F32 = 0,
  CAST_CPU_PRECISION_F64 = 1,
} cast_cpu_precision_t;

typedef enum cast_cpu_simd_width_t {
  CAST_CPU_SIMD_WIDTH_W128 = 128,
  CAST_CPU_SIMD_WIDTH_W256 = 256,
  CAST_CPU_SIMD_WIDTH_W512 = 512,
} cast_cpu_simd_width_t;

typedef enum cast_cpu_matrix_load_mode_t {
  CAST_CPU_MATRIX_LOAD_IMM_VALUE = 0,
  CAST_CPU_MATRIX_LOAD_STACK_LOAD = 1,
} cast_cpu_matrix_load_mode_t;

typedef struct cast_cpu_kernel_gen_spec_t {
  cast_cpu_precision_t precision;
  cast_cpu_simd_width_t simd_width;
  cast_cpu_matrix_load_mode_t mode;
  double ztol;
  double otol;
} cast_cpu_kernel_gen_spec_t;

typedef struct cast_cpu_complex64_t {
  double re;
  double im;
} cast_cpu_complex64_t;

typedef struct cast_cpu_kernel_metadata_t {
  cast_cpu_kernel_id_t kernel_id;
  cast_cpu_precision_t precision;
  cast_cpu_simd_width_t simd_width;
  cast_cpu_matrix_load_mode_t mode;
  uint32_t n_gate_qubits;
} cast_cpu_kernel_metadata_t;

/// Per-kernel data returned by cast_cpu_kernel_generator_finish.
/// The `matrix` and `asm_text` fields are heap-allocated (malloc); call
/// cast_cpu_jit_kernel_records_free to release them after copying the data.
typedef struct cast_cpu_jit_kernel_record_t {
  cast_cpu_kernel_metadata_t metadata;
  void (*entry)(void *);        ///< JIT-compiled function pointer.
  cast_cpu_complex64_t *matrix; ///< NULL for ImmValue mode.
  size_t matrix_len;
  char *asm_text; ///< NULL if request_asm was not called.
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

/// Error handling: returns non-zero on failure and writes a message to err_buf.
int cast_cpu_kernel_generator_generate(cast_cpu_kernel_generator_t *generator,
                                       const cast_cpu_kernel_gen_spec_t *spec,
                                       const cast_cpu_complex64_t *matrix, size_t matrix_len,
                                       const uint32_t *qubits, size_t n_qubits,
                                       cast_cpu_kernel_id_t *out_kernel_id, char *err_buf,
                                       size_t err_buf_len);

// -- Diagnostics --

/// Error handling: returns non-zero on failure and writes a message to err_buf.
int cast_cpu_kernel_generator_request_asm(cast_cpu_kernel_generator_t *generator,
                                          cast_cpu_kernel_id_t kernel_id, char *err_buf,
                                          size_t err_buf_len);

/// Error handling: returns non-zero on failure and writes a message to err_buf.
int cast_cpu_kernel_generator_emit_ir(cast_cpu_kernel_generator_t *generator,
                                      cast_cpu_kernel_id_t kernel_id, char *out_ir,
                                      size_t ir_buf_len, size_t *out_ir_len, char *err_buf,
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
//
// These types are used by the JIT dispatch code within C++ and have matching
// Rust-side #[repr(C)] structs (CastCpuLaunchArgs), but the C typedefs
// themselves are not referenced through the Rust FFI module.

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

#endif // CAST_SIMULATOR_SRC_CPP_CPU_H
