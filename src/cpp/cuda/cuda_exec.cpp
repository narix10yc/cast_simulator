#include "../include/cast_cuda.h"
#include "cuda_util.h"

#include <cuda.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

using namespace cast_cuda_detail;

// ── CUDA driver error helpers ─────────────────────────────────────────────────

static std::string cu_error_string(CUresult rc) {
  const char *name = nullptr;
  const char *desc = nullptr;
  cuGetErrorName(rc, &name);
  cuGetErrorString(rc, &desc);
  std::string s = "CUDA ";
  if (name)
    s += name;
  if (desc) {
    s += " — ";
    s += desc;
  }
  return s;
}

static bool cu_check(CUresult rc, const char *op, std::string &err_out) {
  if (rc == CUDA_SUCCESS)
    return true;
  err_out = std::string(op) + " failed: " + cu_error_string(rc);
  return false;
}

// ── Per-process CUDA initialisation ──────────────────────────────────────────

static CUdevice g_device = 0;
static CUcontext g_context = nullptr;
static std::string g_init_error;

static void cuda_init_once() {
  std::string err;
  CUresult rc = cuInit(0);
  if (!cu_check(rc, "cuInit", err)) {
    g_init_error = err;
    return;
  }

  rc = cuDeviceGet(&g_device, 0);
  if (!cu_check(rc, "cuDeviceGet", err)) {
    g_init_error = err;
    return;
  }

  rc = cuDevicePrimaryCtxRetain(&g_context, g_device);
  if (!cu_check(rc, "cuDevicePrimaryCtxRetain", err)) {
    g_init_error = err;
    return;
  }

  rc = cuCtxSetCurrent(g_context);
  if (!cu_check(rc, "cuCtxSetCurrent", err)) {
    g_init_error = err;
    g_context = nullptr;
  }
}

static bool ensure_cuda(std::string &err) {
  static std::once_flag flag;
  std::call_once(flag, cuda_init_once);
  if (!g_init_error.empty()) {
    err = g_init_error;
    return false;
  }
  if (!g_context) {
    err = "CUDA context not initialized";
    return false;
  }
  // The primary context must be made current for the *calling* thread.
  CUresult const rc = cuCtxSetCurrent(g_context);
  return cu_check(rc, "cuCtxSetCurrent", err);
}

// ── Grid / block helper ───────────────────────────────────────────────────────

static void launch_dims(uint64_t n_combos, uint32_t n_gate_qubits, unsigned int &grid_x,
                        unsigned int &block_x) {
  // 4q+ gates generate large straight-line kernels with high register
  // pressure.  Using block=256 forces the register allocator to spill
  // massively (295 virtual → 48 physical regs on typical 4q kernels).
  // A smaller block gives each thread more of the SM register file,
  // avoiding spills and recovering full bandwidth utilization.
  unsigned int const max_block = (n_gate_qubits >= 4) ? 64 : 256;
  block_x = static_cast<unsigned int>(std::min<uint64_t>(max_block, n_combos));
  if (block_x == 0)
    block_x = 1;
  uint64_t const g = (n_combos + block_x - 1) / block_x;
  grid_x = static_cast<unsigned int>(std::min<uint64_t>(65535, g));
  if (grid_x == 0)
    grid_x = 1;
}

// ── Device capability query ───────────────────────────────────────────────────

extern "C" int cast_cuda_device_sm(uint32_t *out_major, uint32_t *out_minor, char *err_buf,
                                   size_t err_buf_len) {
  std::string err;
  if (!ensure_cuda(err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }
  int major = 0, minor = 0;
  CUresult rc =
      cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, g_device);
  if (!cu_check(rc, "cuDeviceGetAttribute (major)", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }
  rc = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, g_device);
  if (!cu_check(rc, "cuDeviceGetAttribute (minor)", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }
  if (out_major)
    *out_major = static_cast<uint32_t>(major);
  if (out_minor)
    *out_minor = static_cast<uint32_t>(minor);
  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

extern "C" int cast_cuda_free_memory(uint64_t *out_free_bytes, uint64_t *out_total_bytes,
                                     char *err_buf, size_t err_buf_len) {
  std::string err;
  if (!ensure_cuda(err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }
  size_t free_bytes = 0, total_bytes = 0;
  CUresult const rc = cuMemGetInfo(&free_bytes, &total_bytes);
  if (!cu_check(rc, "cuMemGetInfo", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }
  if (out_free_bytes)
    *out_free_bytes = static_cast<uint64_t>(free_bytes);
  if (out_total_bytes)
    *out_total_bytes = static_cast<uint64_t>(total_bytes);
  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

// ── PTX → cubin compilation ───────────────────────────────────────────────────

extern "C" int cast_cuda_ptx_to_cubin(const char *ptx_data, uint8_t **out_cubin,
                                      size_t *out_cubin_len, char *err_buf, size_t err_buf_len) {
  if (!ptx_data || !out_cubin || !out_cubin_len) {
    write_error_message(err_buf, err_buf_len, "arguments must not be null");
    return 1;
  }
  std::string err;
  if (!ensure_cuda(err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }

  // Capture JIT compiler errors so PTX syntax problems surface clearly.
  char jit_log[4096] = {};
  uintptr_t const log_size = sizeof(jit_log);
  CUjit_option opt_keys[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void *opt_vals[] = {static_cast<void *>(jit_log), reinterpret_cast<void *>(log_size)};

  CUlinkState link_state = nullptr;
  CUresult rc = cuLinkCreate(2, opt_keys, opt_vals, &link_state);
  if (!cu_check(rc, "cuLinkCreate", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }

  rc = cuLinkAddData(link_state, CU_JIT_INPUT_PTX,
                     const_cast<void *>(static_cast<const void *>(ptx_data)), std::strlen(ptx_data),
                     "kernel.ptx", 0, nullptr, nullptr);
  if (!cu_check(rc, "cuLinkAddData", err)) {
    if (jit_log[0] != '\0') {
      err += " — ";
      err += jit_log;
    }
    cuLinkDestroy(link_state);
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }

  void *cubin_ptr = nullptr;
  size_t cubin_size = 0;
  rc = cuLinkComplete(link_state, &cubin_ptr, &cubin_size);
  if (!cu_check(rc, "cuLinkComplete", err)) {
    if (jit_log[0] != '\0') {
      err += " — ";
      err += jit_log;
    }
    cuLinkDestroy(link_state);
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }

  // cubin_ptr is owned by the linker state; copy before destroying.
  auto *buf = new uint8_t[cubin_size];
  std::memcpy(buf, cubin_ptr, cubin_size);
  cuLinkDestroy(link_state);

  *out_cubin = buf;
  *out_cubin_len = cubin_size;
  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

extern "C" void cast_cuda_cubin_free(const uint8_t *cubin) { delete[] cubin; }

// ── CUDA module loading ───────────────────────────────────────────────────────

extern "C" void *cast_cuda_cubin_load(const uint8_t *cubin_data, size_t /*cubin_len*/,
                                      char *err_buf, size_t err_buf_len) {
  if (!cubin_data) {
    write_error_message(err_buf, err_buf_len, "cubin_data must not be null");
    return nullptr;
  }
  std::string err;
  if (!ensure_cuda(err)) {
    write_error_message(err_buf, err_buf_len, err);
    return nullptr;
  }
  CUmodule mod = nullptr;
  CUresult const rc = cuModuleLoadData(&mod, cubin_data);
  if (!cu_check(rc, "cuModuleLoadData", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return nullptr;
  }
  clear_error_buffer(err_buf, err_buf_len);
  return static_cast<void *>(mod);
}

extern "C" void cast_cuda_module_unload(void *cu_module) {
  if (cu_module)
    cuModuleUnload(static_cast<CUmodule>(cu_module));
}

extern "C" void *cast_cuda_module_get_function(void *cu_module, const char *func_name,
                                               char *err_buf, size_t err_buf_len) {
  if (!cu_module || !func_name) {
    write_error_message(err_buf, err_buf_len, "cu_module and func_name must not be null");
    return nullptr;
  }
  CUfunction func = nullptr;
  std::string err;
  CUresult const rc = cuModuleGetFunction(&func, static_cast<CUmodule>(cu_module), func_name);
  if (!cu_check(rc, "cuModuleGetFunction", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return nullptr;
  }
  clear_error_buffer(err_buf, err_buf_len);
  return static_cast<void *>(func);
}

// ── CUDA stream ───────────────────────────────────────────────────────────────

extern "C" void *cast_cuda_stream_create(char *err_buf, size_t err_buf_len) {
  std::string err;
  if (!ensure_cuda(err)) {
    write_error_message(err_buf, err_buf_len, err);
    return nullptr;
  }
  CUstream stream = nullptr;
  CUresult const rc = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
  if (!cu_check(rc, "cuStreamCreate", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return nullptr;
  }
  clear_error_buffer(err_buf, err_buf_len);
  return static_cast<void *>(stream);
}

extern "C" void cast_cuda_stream_destroy(void *stream) {
  if (stream)
    cuStreamDestroy(static_cast<CUstream>(stream));
}

extern "C" int cast_cuda_stream_sync(void *stream, char *err_buf, size_t err_buf_len) {
  std::string err;
  CUresult const rc = cuStreamSynchronize(static_cast<CUstream>(stream));
  if (!cu_check(rc, "cuStreamSynchronize", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }
  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

// ── Kernel launch ─────────────────────────────────────────────────────────────

extern "C" int cast_cuda_kernel_launch(void *cu_function, void *stream, uint64_t sv_dptr,
                                       uint32_t n_gate_qubits, uint32_t sv_n_qubits, char *err_buf,
                                       size_t err_buf_len) {
  if (!cu_function) {
    write_error_message(err_buf, err_buf_len, "cu_function must not be null");
    return 1;
  }
  if (sv_n_qubits < n_gate_qubits) {
    write_error_message(err_buf, err_buf_len,
                        "statevector has fewer qubits than the gate kernel requires");
    return 1;
  }

  auto sv_d = static_cast<CUdeviceptr>(sv_dptr);
  CUdeviceptr mat_d = 0;
  uint64_t n_combos = UINT64_C(1) << (sv_n_qubits - n_gate_qubits);

  void *args[] = {&sv_d, &mat_d, &n_combos};

  unsigned int grid_x, block_x;
  launch_dims(n_combos, n_gate_qubits, grid_x, block_x);

  std::string err;
  CUresult const rc =
      cuLaunchKernel(static_cast<CUfunction>(cu_function), grid_x, 1, 1, block_x, 1, 1,
                     /*sharedMemBytes=*/0, static_cast<CUstream>(stream), args, nullptr);
  if (!cu_check(rc, "cuLaunchKernel", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }

  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

// ── CUDA timing events ────────────────────────────────────────────────────────

extern "C" void *cast_cuda_event_create(char *err_buf, size_t err_buf_len) {
  std::string err;
  if (!ensure_cuda(err)) {
    write_error_message(err_buf, err_buf_len, err);
    return nullptr;
  }
  CUevent ev = nullptr;
  CUresult const rc = cuEventCreate(&ev, CU_EVENT_DEFAULT);
  if (!cu_check(rc, "cuEventCreate", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return nullptr;
  }
  clear_error_buffer(err_buf, err_buf_len);
  return static_cast<void *>(ev);
}

extern "C" int cast_cuda_event_record(void *event, void *stream, char *err_buf,
                                      size_t err_buf_len) {
  if (!event || !stream) {
    write_error_message(err_buf, err_buf_len, "event and stream must not be null");
    return 1;
  }
  std::string err;
  CUresult const rc = cuEventRecord(static_cast<CUevent>(event), static_cast<CUstream>(stream));
  if (!cu_check(rc, "cuEventRecord", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }
  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

extern "C" int cast_cuda_event_synchronize(void *event, char *err_buf, size_t err_buf_len) {
  if (!event) {
    write_error_message(err_buf, err_buf_len, "event must not be null");
    return 1;
  }
  std::string err;
  CUresult const rc = cuEventSynchronize(static_cast<CUevent>(event));
  if (!cu_check(rc, "cuEventSynchronize", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }
  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

extern "C" void cast_cuda_event_destroy(void *event) {
  if (event)
    cuEventDestroy(static_cast<CUevent>(event));
}

extern "C" int cast_cuda_event_elapsed_ms(void *start_event, void *end_event, float *out_ms,
                                          char *err_buf, size_t err_buf_len) {
  if (!start_event || !end_event || !out_ms) {
    write_error_message(err_buf, err_buf_len, "arguments must not be null");
    return 1;
  }
  std::string err;
  CUresult const rc = cuEventElapsedTime(out_ms, static_cast<CUevent>(start_event),
                                         static_cast<CUevent>(end_event));
  if (!cu_check(rc, "cuEventElapsedTime", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }
  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

// ── Device-to-device async memcpy ─────────────────────────────────────────────

extern "C" int cast_cuda_memcpy_dtod_async(uint64_t dst, uint64_t src, size_t n_bytes, void *stream,
                                           char *err_buf, size_t err_buf_len) {
  std::string err;
  if (!ensure_cuda(err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }
  CUresult const rc =
      cuMemcpyDtoDAsync(static_cast<CUdeviceptr>(dst), static_cast<CUdeviceptr>(src), n_bytes,
                        static_cast<CUstream>(stream));
  if (!cu_check(rc, "cuMemcpyDtoDAsync", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }
  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

// ── Stateless device memory ───────────────────────────────────────────────────

extern "C" uint64_t cast_cuda_sv_alloc(size_t n_elements, uint8_t precision, char *err_buf,
                                       size_t err_buf_len) {
  std::string err;
  if (!ensure_cuda(err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 0;
  }
  size_t const scalar_bytes = (precision == 0) ? sizeof(float) : sizeof(double);
  CUdeviceptr dptr = 0;
  CUresult const rc = cuMemAlloc(&dptr, n_elements * scalar_bytes);
  if (!cu_check(rc, "cuMemAlloc", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 0;
  }
  clear_error_buffer(err_buf, err_buf_len);
  return static_cast<uint64_t>(dptr);
}

extern "C" void cast_cuda_sv_free(uint64_t dptr) {
  if (dptr)
    cuMemFree(static_cast<CUdeviceptr>(dptr));
}

extern "C" int cast_cuda_sv_zero(uint64_t dptr, size_t n_elements, uint8_t precision, char *err_buf,
                                 size_t err_buf_len) {
  auto const d = static_cast<CUdeviceptr>(dptr);
  size_t const scalar_bytes = (precision == 0) ? sizeof(float) : sizeof(double);

  std::string err;
  CUresult rc = cuMemsetD8(d, 0, n_elements * scalar_bytes);
  if (!cu_check(rc, "cuMemsetD8", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }

  // Write 1.0 into the real part of amplitude[0].
  if (precision == 0) {
    float one = 1.0f;
    rc = cuMemcpyHtoD(d, &one, sizeof(float));
  } else {
    double one = 1.0;
    rc = cuMemcpyHtoD(d, &one, sizeof(double));
  }
  if (!cu_check(rc, "cuMemcpyHtoD (zero state)", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }

  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

extern "C" int cast_cuda_sv_upload(uint64_t dptr, const double *host_data, size_t n_elements,
                                   uint8_t precision, char *err_buf, size_t err_buf_len) {
  auto const d = static_cast<CUdeviceptr>(dptr);
  std::string err;
  CUresult rc;

  if (precision == 1) { // F64: direct copy
    rc = cuMemcpyHtoD(d, host_data, n_elements * sizeof(double));
    if (!cu_check(rc, "cuMemcpyHtoD", err)) {
      write_error_message(err_buf, err_buf_len, err);
      return 1;
    }
  } else { // F32: narrow on host then upload
    std::vector<float> fdata(n_elements);
    for (size_t i = 0; i < n_elements; ++i)
      fdata[i] = static_cast<float>(host_data[i]);
    rc = cuMemcpyHtoD(d, fdata.data(), n_elements * sizeof(float));
    if (!cu_check(rc, "cuMemcpyHtoD (f32)", err)) {
      write_error_message(err_buf, err_buf_len, err);
      return 1;
    }
  }

  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

extern "C" int cast_cuda_sv_download(uint64_t dptr, double *host_data, size_t n_elements,
                                     uint8_t precision, char *err_buf, size_t err_buf_len) {
  auto const d = static_cast<CUdeviceptr>(dptr);
  std::string err;
  CUresult rc;

  if (precision == 1) { // F64: direct copy
    rc = cuMemcpyDtoH(host_data, d, n_elements * sizeof(double));
    if (!cu_check(rc, "cuMemcpyDtoH", err)) {
      write_error_message(err_buf, err_buf_len, err);
      return 1;
    }
  } else { // F32: download then widen to f64
    std::vector<float> fdata(n_elements);
    rc = cuMemcpyDtoH(fdata.data(), d, n_elements * sizeof(float));
    if (!cu_check(rc, "cuMemcpyDtoH (f32)", err)) {
      write_error_message(err_buf, err_buf_len, err);
      return 1;
    }
    for (size_t i = 0; i < n_elements; ++i)
      host_data[i] = static_cast<double>(fdata[i]);
  }

  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}
