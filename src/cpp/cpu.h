#ifndef CAST_SIMULATOR_SRC_CPP_CPU_H
#define CAST_SIMULATOR_SRC_CPP_CPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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

/// cast::CPUKernelGenConfig
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

typedef struct cast_cpu_kernel_generator_t cast_cpu_kernel_generator_t;
typedef struct cast_cpu_jit_session_t cast_cpu_jit_session_t;

cast_cpu_kernel_generator_t *cast_cpu_kernel_generator_new(void);
void cast_cpu_kernel_generator_delete(cast_cpu_kernel_generator_t *generator);

// Returns 0 on success. On failure, returns non-zero and writes a
// human-readable message into err_buf if provided.
int cast_cpu_kernel_generator_generate(cast_cpu_kernel_generator_t *generator,
                                       const cast_cpu_kernel_gen_spec_t *spec,
                                       const cast_cpu_complex64_t *matrix, size_t matrix_len,
                                       const uint32_t *qubits, size_t n_qubits,
                                       cast_cpu_kernel_id_t *out_kernel_id, char *err_buf,
                                       size_t err_buf_len);

// Marks the kernel identified by kernel_id for assembly capture during init_jit.
// Call this before cast_cpu_kernel_generator_finish.
// Returns 0 on success and non-zero if kernel_id is not found.
int cast_cpu_kernel_generator_request_asm(cast_cpu_kernel_generator_t *generator,
                                          cast_cpu_kernel_id_t kernel_id, char *err_buf,
                                          size_t err_buf_len);

// Runs the O1 pass pipeline on the kernel identified by kernel_id and returns
// its optimized LLVM IR as a null-terminated string.
//
// Two-call pattern for the buffer:
//   1. Pass out_ir = NULL to query the required buffer size via *out_ir_len.
//   2. Allocate out_ir_len + 1 bytes, then call again with out_ir pointing to
//      that buffer and ir_buf_len set to the allocated size.
//
// out_ir_len may be NULL if you only need the text and provide a large enough
// buffer.
// Returns 0 on success and non-zero on error (message written to err_buf).
int cast_cpu_kernel_generator_emit_ir(cast_cpu_kernel_generator_t *generator,
                                      cast_cpu_kernel_id_t kernel_id, char *out_ir,
                                      size_t ir_buf_len, size_t *out_ir_len, char *err_buf,
                                      size_t err_buf_len);

// Compiles all kernels previously registered with cast_cpu_kernel_generator_generate and
// produces a JIT session.
// On success, returns 0, writes the session into *out_session, and deletes the generator.
// Ownership is transferred; the caller must not use or free the generator afterwards.
// On failure, returns non-zero and leaves the generator intact.
int cast_cpu_kernel_generator_finish(cast_cpu_kernel_generator_t *generator,
                                     cast_cpu_jit_session_t **out_session, char *err_buf,
                                     size_t err_buf_len);

int cast_cpu_jit_session_apply(cast_cpu_jit_session_t *session, cast_cpu_kernel_id_t kernel_id,
                               void *sv, uint32_t n_qubits, cast_cpu_precision_t sv_precision,
                               cast_cpu_simd_width_t sv_simd_width, int32_t n_threads,
                               char *err_buf, size_t err_buf_len);

// Returns the native assembly text emitted during JIT compilation for the
// kernel identified by kernel_id.
//
// Same two-call pattern as cast_cpu_kernel_generator_emit_ir:
//   1. Pass out_asm = NULL to query the length via *out_asm_len.
//   2. Allocate out_asm_len + 1 bytes, then call again with the buffer.
//
// Returns 0 on success and non-zero on error (message written to err_buf).
int cast_cpu_jit_session_emit_asm(cast_cpu_jit_session_t *session, cast_cpu_kernel_id_t kernel_id,
                                  char *out_asm, size_t asm_buf_len, size_t *out_asm_len,
                                  char *err_buf, size_t err_buf_len);

void cast_cpu_jit_session_delete(cast_cpu_jit_session_t *session);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CAST_SIMULATOR_SRC_CPP_CPU_H
