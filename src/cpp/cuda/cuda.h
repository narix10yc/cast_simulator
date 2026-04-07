#ifndef CAST_SIMULATOR_SRC_CPP_CUDA_H
#define CAST_SIMULATOR_SRC_CPP_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Exported to Rust ────────────────────────────────────────────────────────
//
// Every type and function below has a matching Rust declaration:
//   - Types     → #[repr(C)] structs/enums in src/cuda/types.rs or src/cuda/mod.rs::mod ffi
//   - Functions → unsafe extern "C" in src/cuda/mod.rs::mod ffi
//                 or src/cuda/statevector.rs::mod ffi
//
// Functions guarded by the `cuda` Cargo feature are noted inline.

typedef enum cast_cuda_precision_t {
  CAST_CUDA_PRECISION_F32 = 0,
  CAST_CUDA_PRECISION_F64 = 1,
} cast_cuda_precision_t;

typedef struct cast_cuda_kernel_gen_spec_t {
  cast_cuda_precision_t precision;
  double ztol;
  double otol;
  uint32_t sm_major;
  uint32_t sm_minor;
} cast_cuda_kernel_gen_spec_t;

typedef struct cast_cuda_complex64_t {
  double re;
  double im;
} cast_cuda_complex64_t;

// -- Gate PTX compilation --

int cast_cuda_compile_gate_ptx(const cast_cuda_kernel_gen_spec_t *spec,
                               const cast_cuda_complex64_t *matrix, size_t matrix_len,
                               const uint32_t *qubits, size_t n_qubits, char **out_ptx,
                               char **out_func_name, uint32_t *out_n_gate_qubits,
                               uint8_t *out_precision, char *err_buf, size_t err_buf_len);
void cast_cuda_str_free(char *s);

// -- Device capability query (cuda feature) --

int cast_cuda_device_sm(uint32_t *out_major, uint32_t *out_minor, char *err_buf,
                        size_t err_buf_len);
int cast_cuda_free_memory(uint64_t *out_free_bytes, uint64_t *out_total_bytes, char *err_buf,
                          size_t err_buf_len);

// -- CUDA stream (cuda feature) --

void *cast_cuda_stream_create(char *err_buf, size_t err_buf_len);
void cast_cuda_stream_destroy(void *stream);
int cast_cuda_stream_sync(void *stream, char *err_buf, size_t err_buf_len);

// -- PTX → cubin JIT compilation (cuda feature) --

int cast_cuda_ptx_to_cubin(const char *ptx_data, uint8_t **out_cubin, size_t *out_cubin_len,
                           char *err_buf, size_t err_buf_len);
void cast_cuda_cubin_free(uint8_t *cubin);

// -- Module loading and kernel launch (cuda feature) --

void *cast_cuda_cubin_load(const uint8_t *cubin_data, size_t cubin_len, char *err_buf,
                           size_t err_buf_len);
void cast_cuda_module_unload(void *cu_module);
void *cast_cuda_module_get_function(void *cu_module, const char *func_name, char *err_buf,
                                    size_t err_buf_len);
int cast_cuda_kernel_launch(void *cu_function, void *stream, uint64_t sv_dptr,
                            uint32_t n_gate_qubits, uint32_t sv_n_qubits, uint8_t precision,
                            char *err_buf, size_t err_buf_len);

// -- CUDA timing events (cuda feature) --

void *cast_cuda_event_create(char *err_buf, size_t err_buf_len);
int cast_cuda_event_record(void *event, void *stream, char *err_buf, size_t err_buf_len);
int cast_cuda_event_synchronize(void *event, char *err_buf, size_t err_buf_len);
void cast_cuda_event_destroy(void *event);
int cast_cuda_event_elapsed_ms(void *start_event, void *end_event, float *out_ms, char *err_buf,
                               size_t err_buf_len);

// -- Device-to-device async memcpy (cuda feature) --

int cast_cuda_memcpy_dtod_async(uint64_t dst, uint64_t src, size_t n_bytes, void *stream,
                                char *err_buf, size_t err_buf_len);

// -- Device memory / statevector (cuda feature) --

uint64_t cast_cuda_sv_alloc(size_t n_elements, uint8_t precision, char *err_buf,
                            size_t err_buf_len);
void cast_cuda_sv_free(uint64_t dptr);
int cast_cuda_sv_zero(uint64_t dptr, size_t n_elements, uint8_t precision, char *err_buf,
                      size_t err_buf_len);
int cast_cuda_sv_upload(uint64_t dptr, const double *host_data, size_t n_elements,
                        uint8_t precision, char *err_buf, size_t err_buf_len);
int cast_cuda_sv_download(uint64_t dptr, double *host_data, size_t n_elements, uint8_t precision,
                          char *err_buf, size_t err_buf_len);

// -- Device-side reductions (cuda feature) --

/// Compute sum of squares of all n_elements scalars at dptr.
/// precision: 0 = F32, 1 = F64.  Result is always returned as f64.
int cast_cuda_norm_squared(uint64_t dptr, size_t n_elements, uint8_t precision, double *out_norm_sq,
                           char *err_buf, size_t err_buf_len);

/// Scale all n_elements scalars at dptr by `scale`.
int cast_cuda_scale(uint64_t dptr, size_t n_elements, uint8_t precision, double scale,
                    char *err_buf, size_t err_buf_len);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CAST_SIMULATOR_SRC_CPP_CUDA_H
