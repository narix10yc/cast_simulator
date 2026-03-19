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

/// Function type for JIT-compiled kernel entry points.
typedef void cast_cpu_kernel_entry_t(void *);

/// Arguments passed to each JIT kernel invocation.
typedef struct cast_cpu_launch_args_t {
  void *sv;
  uint64_t ctr_begin;
  uint64_t ctr_end;
  void *p_mat;
} cast_cpu_launch_args_t;

/// Per-kernel data returned by cast_cpu_kernel_generator_finish.
/// The `matrix` and `asm_text` fields are heap-allocated (malloc); call
/// cast_cpu_jit_kernel_records_free to release them after copying the data.
typedef struct cast_cpu_jit_kernel_record_t {
  cast_cpu_kernel_metadata_t metadata;
  cast_cpu_kernel_entry_t *entry; ///< JIT-compiled function pointer.
  cast_cpu_complex64_t *matrix;   ///< NULL for ImmValue mode.
  size_t matrix_len;
  char *asm_text; ///< NULL if request_asm was not called before finish.
} cast_cpu_jit_kernel_record_t;

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

// Marks the kernel identified by kernel_id for assembly capture during finish.
// Call this before cast_cpu_kernel_generator_finish.
// Returns 0 on success and non-zero if kernel_id is not found.
int cast_cpu_kernel_generator_request_asm(cast_cpu_kernel_generator_t *generator,
                                          cast_cpu_kernel_id_t kernel_id, char *err_buf,
                                          size_t err_buf_len);

// Runs the O1 pass pipeline on the kernel identified by kernel_id and returns
// its optimized LLVM IR as a null-terminated string (two-call pattern).
// Returns 0 on success and non-zero on error (message written to err_buf).
int cast_cpu_kernel_generator_emit_ir(cast_cpu_kernel_generator_t *generator,
                                      cast_cpu_kernel_id_t kernel_id, char *out_ir,
                                      size_t ir_buf_len, size_t *out_ir_len, char *err_buf,
                                      size_t err_buf_len);

// Compiles all kernels and produces a JIT session plus a malloc'd array of
// per-kernel records.
//
// On success: writes the session to *out_session, the malloc'd records array
// to *out_records, the record count to *out_n_records, deletes the generator,
// and returns 0. The caller must call cast_cpu_jit_kernel_records_free after
// copying the data out of the records.
//
// On failure: returns non-zero, leaves the generator intact, and writes an
// error message to err_buf.
int cast_cpu_kernel_generator_finish(cast_cpu_kernel_generator_t *generator,
                                     cast_cpu_jit_session_t **out_session,
                                     cast_cpu_jit_kernel_record_t **out_records,
                                     size_t *out_n_records, char *err_buf, size_t err_buf_len);

// Frees the matrix and asm_text fields of each record and then the records
// array itself. Safe to call with records=NULL or n=0.
void cast_cpu_jit_kernel_records_free(cast_cpu_jit_kernel_record_t *records, size_t n);

void cast_cpu_jit_session_delete(cast_cpu_jit_session_t *session);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CAST_SIMULATOR_SRC_CPP_CPU_H
