#ifndef CAST_SIMULATOR_SRC_CPP_CUDA_H
#define CAST_SIMULATOR_SRC_CPP_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cast_cuda_precision_t {
  CAST_CUDA_PRECISION_F32 = 0,
  CAST_CUDA_PRECISION_F64 = 1,
} cast_cuda_precision_t;

/// Configuration for kernel code generation.
/// sm_major / sm_minor specify the target GPU compute capability (e.g. 8, 6
/// for sm_86). They are explicit so PTX compilation can proceed without a
/// live CUDA device.
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

// ── Gate PTX compilation ──────────────────────────────────────────────────────

/// Compile a gate matrix directly to PTX in one call.
///
/// On success: returns 0 and writes C-string pointers into *out_ptx and
/// *out_func_name (both heap-allocated; caller must free each with
/// cast_cuda_str_free), the gate qubit count into *out_n_gate_qubits, and
/// the precision byte into *out_precision.
/// On failure: returns non-zero and writes a message into err_buf.
int cast_cuda_compile_gate_ptx(const cast_cuda_kernel_gen_spec_t *spec,
                               const cast_cuda_complex64_t *matrix, size_t matrix_len,
                               const uint32_t *qubits, size_t n_qubits, char **out_ptx,
                               char **out_func_name, uint32_t *out_n_gate_qubits,
                               uint8_t *out_precision, char *err_buf, size_t err_buf_len);

/// Free a string returned by cast_cuda_compile_gate_ptx.
void cast_cuda_str_free(char *s);

// ── Device capability query ───────────────────────────────────────────────────

/// Query the compute capability of device 0.
/// Writes major/minor into *out_major / *out_minor.
/// Returns 0 on success; writes a message into err_buf on failure.
int cast_cuda_device_sm(uint32_t *out_major, uint32_t *out_minor, char *err_buf,
                        size_t err_buf_len);

// ── CUDA stream ───────────────────────────────────────────────────────────────

/// Create a new CUDA stream. Returns an opaque CUstream as void*, or NULL on
/// failure (message written to err_buf). Initialises the CUDA driver on first call.
void *cast_cuda_stream_create(char *err_buf, size_t err_buf_len);

void cast_cuda_stream_destroy(void *stream);

/// Block the calling thread until all work enqueued on the stream completes.
int cast_cuda_stream_sync(void *stream, char *err_buf, size_t err_buf_len);

// ── PTX → cubin compilation ───────────────────────────────────────────────────

/// Compile PTX to device-native cubin using the CUDA JIT linker.
///
/// On success: returns 0, allocates a buffer at *out_cubin (caller must free
/// it with cast_cuda_cubin_free), and writes the byte count into *out_cubin_len.
/// On failure: returns non-zero and writes a message into err_buf; any JIT
/// compiler error log is appended to the message.
int cast_cuda_ptx_to_cubin(const char *ptx_data, uint8_t **out_cubin, size_t *out_cubin_len,
                           char *err_buf, size_t err_buf_len);

/// Free a cubin buffer returned by cast_cuda_ptx_to_cubin.
void cast_cuda_cubin_free(uint8_t *cubin);

// ── CUDA module loading ───────────────────────────────────────────────────────

void cast_cuda_module_unload(void *cu_module);

/// Load cubin bytes into a CUDA driver module. Returns an opaque CUmodule as
/// void*, or NULL on failure (message written to err_buf).
void *cast_cuda_cubin_load(const uint8_t *cubin_data, size_t cubin_len, char *err_buf,
                           size_t err_buf_len);

/// Resolve a kernel entry point from a loaded module. Returns an opaque
/// CUfunction as void*, or NULL on failure.
void *cast_cuda_module_get_function(void *cu_module, const char *func_name, char *err_buf,
                                    size_t err_buf_len);

/// Enqueue a kernel launch on `stream` without synchronising.
/// precision: 0 = F32, 1 = F64.
int cast_cuda_kernel_launch(void *cu_function, void *stream, uint64_t sv_dptr,
                            uint32_t n_gate_qubits, uint32_t sv_n_qubits, uint8_t precision,
                            char *err_buf, size_t err_buf_len);

// ── CUDA timing events ───────────────────────────────────────────────────────

/// Create a CUDA timing event. Returns an opaque CUevent as void*, or NULL on
/// failure (message written to err_buf).
void *cast_cuda_event_create(char *err_buf, size_t err_buf_len);

/// Record `event` at the current position on `stream` (non-blocking).
/// Returns 0 on success; writes a message into err_buf on failure.
int cast_cuda_event_record(void *event, void *stream, char *err_buf, size_t err_buf_len);

/// Destroy a CUDA event previously created with cast_cuda_event_create.
void cast_cuda_event_destroy(void *event);

/// Query the elapsed GPU time in milliseconds between two recorded events.
/// Both events must have fired (i.e., the stream must have been synchronised).
/// Writes the elapsed time into *out_ms.
/// Returns 0 on success; writes a message into err_buf on failure.
int cast_cuda_event_elapsed_ms(void *start_event, void *end_event, float *out_ms,
                               char *err_buf, size_t err_buf_len);

// ── Device-to-device async memcpy ────────────────────────────────────────────

/// Enqueue an asynchronous device-to-device copy of `n_bytes` bytes from `src`
/// to `dst` on `stream`.  Returns 0 on success; writes a message into err_buf
/// on failure.
int cast_cuda_memcpy_dtod_async(uint64_t dst, uint64_t src, size_t n_bytes, void *stream,
                                char *err_buf, size_t err_buf_len);

// ── Stateless device memory ───────────────────────────────────────────────────

/// Allocate device memory for n_elements scalars of the given precision.
/// Returns the CUdeviceptr as uint64_t (0 on failure).
uint64_t cast_cuda_sv_alloc(size_t n_elements, uint8_t precision, char *err_buf,
                            size_t err_buf_len);

void cast_cuda_sv_free(uint64_t dptr);

/// Set all bytes to zero then write 1.0 into amplitude[0].re (|0⟩ state).
int cast_cuda_sv_zero(uint64_t dptr, size_t n_elements, uint8_t precision, char *err_buf,
                      size_t err_buf_len);

/// Upload from host f64 buffer; narrows to f32 if precision == 0.
int cast_cuda_sv_upload(uint64_t dptr, const double *host_data, size_t n_elements,
                        uint8_t precision, char *err_buf, size_t err_buf_len);

/// Download to host f64 buffer; widens from f32 if precision == 0.
int cast_cuda_sv_download(uint64_t dptr, double *host_data, size_t n_elements, uint8_t precision,
                          char *err_buf, size_t err_buf_len);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CAST_SIMULATOR_SRC_CPP_CUDA_H
