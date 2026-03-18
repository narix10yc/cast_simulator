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
  double ztol; ///< threshold below which a matrix element is treated as 0
  double otol; ///< threshold within which a matrix element is treated as ±1
  uint32_t sm_major;
  uint32_t sm_minor;
} cast_cuda_kernel_gen_spec_t;

typedef struct cast_cuda_complex64_t {
  double re;
  double im;
} cast_cuda_complex64_t;

typedef struct cast_cuda_kernel_generator_t cast_cuda_kernel_generator_t;
typedef struct cast_cuda_kernel_artifacts_t cast_cuda_kernel_artifacts_t;

// ── Generator ─────────────────────────────────────────────────────────────────

cast_cuda_kernel_generator_t *cast_cuda_kernel_generator_new(void);
void cast_cuda_kernel_generator_delete(cast_cuda_kernel_generator_t *generator);

/// Generates LLVM NVPTX IR for one gate kernel.
/// Returns 0 on success; writes a human-readable message into err_buf on failure.
int cast_cuda_kernel_generator_generate(cast_cuda_kernel_generator_t *generator,
                                        const cast_cuda_kernel_gen_spec_t *spec,
                                        const cast_cuda_complex64_t *matrix,
                                        size_t matrix_len,
                                        const uint32_t *qubits, size_t n_qubits,
                                        cast_cuda_kernel_id_t *out_kernel_id,
                                        char *err_buf, size_t err_buf_len);

/// Returns the optimized NVPTX LLVM IR for the kernel as a null-terminated string.
///
/// Two-call pattern:
///   1. Pass out_ir = NULL to query the required length via *out_ir_len.
///   2. Allocate out_ir_len + 1 bytes, then call again with the buffer.
int cast_cuda_kernel_generator_emit_ir(cast_cuda_kernel_generator_t *generator,
                                       cast_cuda_kernel_id_t kernel_id,
                                       char *out_ir, size_t ir_buf_len,
                                       size_t *out_ir_len,
                                       char *err_buf, size_t err_buf_len);

/// Compiles all kernels (optimize → PTX → cubin) and produces a compilation
/// session.  On success: returns 0, writes the session into *out_session, and
/// deletes the generator.  On failure: returns non-zero, leaves the generator
/// intact.
int cast_cuda_kernel_generator_finish(cast_cuda_kernel_generator_t *generator,
                                      cast_cuda_kernel_artifacts_t **out_session,
                                      char *err_buf, size_t err_buf_len);

// ── Compilation session — indexed accessors ───────────────────────────────────
//
// These allow the Rust layer to drain all compiled kernel data in one pass
// and then delete the session, so the C++ object does not need to outlive
// the caller.

uint32_t cast_cuda_kernel_artifacts_n_kernels(
    const cast_cuda_kernel_artifacts_t *session);

cast_cuda_kernel_id_t cast_cuda_kernel_artifacts_kernel_id_at(
    const cast_cuda_kernel_artifacts_t *session, uint32_t idx);

uint32_t cast_cuda_kernel_artifacts_n_gate_qubits_at(
    const cast_cuda_kernel_artifacts_t *session, uint32_t idx);

/// Returns 0 for F32, 1 for F64.
uint8_t cast_cuda_kernel_artifacts_precision_at(
    const cast_cuda_kernel_artifacts_t *session, uint32_t idx);

/// Returns a pointer into session-owned storage; valid until the session is deleted.
const char *cast_cuda_kernel_artifacts_func_name_at(
    const cast_cuda_kernel_artifacts_t *session, uint32_t idx);

/// Two-call pattern (pass out_ptx = NULL to size, then fill).
int cast_cuda_kernel_artifacts_emit_ptx(cast_cuda_kernel_artifacts_t *session,
                                           cast_cuda_kernel_id_t kernel_id,
                                           char *out_ptx, size_t ptx_buf_len,
                                           size_t *out_ptx_len,
                                           char *err_buf, size_t err_buf_len);

/// Two-call pattern (pass out_cubin = NULL to size, then fill).
int cast_cuda_kernel_artifacts_emit_cubin(cast_cuda_kernel_artifacts_t *session,
                                             cast_cuda_kernel_id_t kernel_id,
                                             uint8_t *out_cubin,
                                             size_t cubin_buf_len,
                                             size_t *out_cubin_len,
                                             char *err_buf, size_t err_buf_len);

void cast_cuda_kernel_artifacts_delete(cast_cuda_kernel_artifacts_t *session);

// ── Device capability query ───────────────────────────────────────────────────

/// Query the compute capability of device 0.
/// Writes major/minor into *out_major / *out_minor.
/// Returns 0 on success; writes a message into err_buf on failure.
int cast_cuda_device_sm(uint32_t *out_major, uint32_t *out_minor,
                        char *err_buf, size_t err_buf_len);

// ── Stateless CUDA module loading ─────────────────────────────────────────────
//
// The exec session is now managed on the Rust side. C++ exposes three
// stateless calls; Rust holds the resulting CUmodule/CUfunction handles
// as opaque void pointers.

/// Load PTX into a CUDA driver module. Returns an opaque CUmodule as void*,
/// or NULL on failure (message written to err_buf).
void *cast_cuda_ptx_load(const char *ptx_data,
                         char *err_buf, size_t err_buf_len);

void cast_cuda_module_unload(void *cu_module);

/// Resolve a kernel entry point from a loaded module. Returns an opaque
/// CUfunction as void*, or NULL on failure.
void *cast_cuda_module_get_function(void *cu_module, const char *func_name,
                                    char *err_buf, size_t err_buf_len);

/// Launch the kernel on the device statevector and synchronize.
/// precision: 0 = F32, 1 = F64.
int cast_cuda_kernel_apply(void *cu_function,
                           uint64_t sv_dptr,
                           uint32_t n_gate_qubits, uint32_t sv_n_qubits,
                           uint8_t precision,
                           char *err_buf, size_t err_buf_len);

// ── Stateless device memory ───────────────────────────────────────────────────
//
// The statevector is now managed on the Rust side. Rust holds the CUdeviceptr
// as a uint64_t; C++ provides thin wrappers around cuMem* calls.

/// Allocate device memory for n_elements scalars of the given precision.
/// Returns the CUdeviceptr as uint64_t (0 on failure).
uint64_t cast_cuda_sv_alloc(size_t n_elements, uint8_t precision,
                            char *err_buf, size_t err_buf_len);

void cast_cuda_sv_free(uint64_t dptr);

/// Set all bytes to zero then write 1.0 into amplitude[0].re (|0⟩ state).
int cast_cuda_sv_zero(uint64_t dptr, size_t n_elements, uint8_t precision,
                      char *err_buf, size_t err_buf_len);

/// Upload from host f64 buffer; narrows to f32 if precision == 0.
int cast_cuda_sv_upload(uint64_t dptr, const double *host_data, size_t n_elements,
                        uint8_t precision, char *err_buf, size_t err_buf_len);

/// Download to host f64 buffer; widens from f32 if precision == 0.
int cast_cuda_sv_download(uint64_t dptr, double *host_data, size_t n_elements,
                          uint8_t precision, char *err_buf, size_t err_buf_len);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CAST_SIMULATOR_SRC_CPP_CUDA_H
