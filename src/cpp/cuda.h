#ifndef CAST_SIMULATOR_SRC_CPP_CUDA_H
#define CAST_SIMULATOR_SRC_CPP_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t cast_cuda_kernel_id_t;

typedef enum cast_cuda_precision_t {
  CAST_CUDA_PRECISION_F32 = 0,
  CAST_CUDA_PRECISION_F64 = 1,
} cast_cuda_precision_t;

/// Configuration passed to the kernel generator.
/// sm_major / sm_minor specify the target GPU compute capability (e.g. 8, 6
/// for sm_86). They are explicit here so compilation can proceed without a
/// live CUDA device — the caller queries the device and passes the arch.
typedef struct cast_cuda_kernel_gen_spec_t {
  cast_cuda_precision_t precision;
  double ztol; /// threshold below which a matrix element is treated as 0
  double otol; /// threshold within which a matrix element is treated as ±1
  uint32_t sm_major;
  uint32_t sm_minor;
} cast_cuda_kernel_gen_spec_t;

typedef struct cast_cuda_complex64_t {
  double re;
  double im;
} cast_cuda_complex64_t;

typedef struct cast_cuda_kernel_generator_t cast_cuda_kernel_generator_t;
typedef struct cast_cuda_compilation_session_t cast_cuda_compilation_session_t;

// ── Generator ─────────────────────────────────────────────────────────────────

cast_cuda_kernel_generator_t *cast_cuda_kernel_generator_new(void);
void cast_cuda_kernel_generator_delete(cast_cuda_kernel_generator_t *generator);

/// Generates LLVM NVPTX IR for one gate kernel.
/// Returns 0 on success; writes a human-readable message into err_buf on
/// failure.
int cast_cuda_kernel_generator_generate(cast_cuda_kernel_generator_t *generator,
                                        const cast_cuda_kernel_gen_spec_t *spec,
                                        const cast_cuda_complex64_t *matrix,
                                        size_t matrix_len,
                                        const uint32_t *qubits, size_t n_qubits,
                                        cast_cuda_kernel_id_t *out_kernel_id,
                                        char *err_buf, size_t err_buf_len);

/// Returns the optimized NVPTX LLVM IR for the kernel as a null-terminated
/// string.  Runs O1 with the NVPTX target machine if not already done.
///
/// Two-call pattern:
///   1. Pass out_ir = NULL to query the required length via *out_ir_len.
///   2. Allocate out_ir_len + 1 bytes, then call again with the buffer.
///
/// out_ir_len may be NULL if you supply a large enough buffer on the first
/// call.  Returns 0 on success and non-zero on error.
int cast_cuda_kernel_generator_emit_ir(cast_cuda_kernel_generator_t *generator,
                                       cast_cuda_kernel_id_t kernel_id,
                                       char *out_ir, size_t ir_buf_len,
                                       size_t *out_ir_len,
                                       char *err_buf, size_t err_buf_len);

/// Compiles all kernels (optimize → PTX → cubin) and produces a compilation
/// session.
/// On success, returns 0, writes the session into *out_session, and deletes
/// the generator (ownership transferred).
/// On failure, returns non-zero and leaves the generator intact.
int cast_cuda_kernel_generator_finish(cast_cuda_kernel_generator_t *generator,
                                      cast_cuda_compilation_session_t **out_session,
                                      char *err_buf, size_t err_buf_len);

// ── Compilation session ────────────────────────────────────────────────────────

/// Returns the PTX assembly string for the compiled kernel.
///
/// Two-call pattern (same as emit_ir above).
int cast_cuda_compilation_session_emit_ptx(cast_cuda_compilation_session_t *session,
                                           cast_cuda_kernel_id_t kernel_id,
                                           char *out_ptx, size_t ptx_buf_len,
                                           size_t *out_ptx_len,
                                           char *err_buf, size_t err_buf_len);

/// Returns the cubin binary for the compiled kernel.
///
/// Two-call pattern:
///   1. Pass out_cubin = NULL to query the required size via *out_cubin_len.
///   2. Allocate out_cubin_len bytes, then call again with the buffer.
int cast_cuda_compilation_session_emit_cubin(cast_cuda_compilation_session_t *session,
                                             cast_cuda_kernel_id_t kernel_id,
                                             uint8_t *out_cubin,
                                             size_t cubin_buf_len,
                                             size_t *out_cubin_len,
                                             char *err_buf, size_t err_buf_len);

void cast_cuda_compilation_session_delete(cast_cuda_compilation_session_t *session);

// ── Execution session ──────────────────────────────────────────────────────────
//
// An execution session loads all PTX kernels from a compilation session into
// the CUDA driver as CUmodule objects, ready to launch.  It does NOT require
// the compilation session to outlive it.

typedef struct cast_cuda_exec_session_t cast_cuda_exec_session_t;

/// Creates an execution session from a compilation session.
/// Initialises the CUDA driver (once per process), loads each kernel's PTX
/// into a CUmodule, and resolves the entry-point CUfunction.
/// Returns NULL on failure (message written to err_buf).
cast_cuda_exec_session_t *
cast_cuda_exec_session_new(const cast_cuda_compilation_session_t *session,
                           char *err_buf, size_t err_buf_len);

void cast_cuda_exec_session_delete(cast_cuda_exec_session_t *session);

/// Applies one compiled kernel to a device statevector in-place.
/// Synchronises the device before returning.  Returns 0 on success.
int cast_cuda_exec_session_apply(cast_cuda_exec_session_t *session,
                                 cast_cuda_kernel_id_t kernel_id,
                                 struct cast_cuda_statevector_t *sv,
                                 char *err_buf, size_t err_buf_len);

// ── Device statevector ─────────────────────────────────────────────────────────

typedef struct cast_cuda_statevector_t cast_cuda_statevector_t;

/// Allocates a device statevector for 2^n_qubits complex amplitudes.
/// Returns NULL on failure.
cast_cuda_statevector_t *
cast_cuda_statevector_new(uint32_t n_qubits, cast_cuda_precision_t precision,
                          char *err_buf, size_t err_buf_len);

void cast_cuda_statevector_delete(cast_cuda_statevector_t *sv);

/// Sets the statevector to the |0⟩ computational basis state.
int cast_cuda_statevector_zero(cast_cuda_statevector_t *sv,
                               char *err_buf, size_t err_buf_len);

/// Uploads amplitudes from a host buffer of n_elements doubles, laid out as
/// interleaved (re0, im0, re1, im1, ...).  For F32 statevectors the values are
/// narrowed to float on the host before the transfer.
/// n_elements must equal 2 * 2^n_qubits.
int cast_cuda_statevector_upload(cast_cuda_statevector_t *sv,
                                 const double *host_data, size_t n_elements,
                                 char *err_buf, size_t err_buf_len);

/// Downloads amplitudes to a host buffer of n_elements doubles.
/// n_elements must equal 2 * 2^n_qubits.
int cast_cuda_statevector_download(const cast_cuda_statevector_t *sv,
                                   double *host_data, size_t n_elements,
                                   char *err_buf, size_t err_buf_len);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CAST_SIMULATOR_SRC_CPP_CUDA_H
