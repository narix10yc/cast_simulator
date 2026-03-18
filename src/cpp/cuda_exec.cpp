#include "cuda.h"
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
  if (name) s += name;
  if (desc) { s += " — "; s += desc; }
  return s;
}

static bool cu_check(CUresult rc, const char *op, std::string &err_out) {
  if (rc == CUDA_SUCCESS) return true;
  err_out = std::string(op) + " failed: " + cu_error_string(rc);
  return false;
}

// ── Per-process CUDA initialisation ──────────────────────────────────────────

static CUdevice  g_device  = 0;
static CUcontext g_context = nullptr;
static std::string g_init_error;

static void cuda_init_once() {
  std::string err;
  CUresult rc = cuInit(0);
  if (!cu_check(rc, "cuInit", err)) { g_init_error = err; return; }

  rc = cuDeviceGet(&g_device, 0);
  if (!cu_check(rc, "cuDeviceGet", err)) { g_init_error = err; return; }

  rc = cuDevicePrimaryCtxRetain(&g_context, g_device);
  if (!cu_check(rc, "cuDevicePrimaryCtxRetain", err)) { g_init_error = err; return; }

  rc = cuCtxSetCurrent(g_context);
  if (!cu_check(rc, "cuCtxSetCurrent", err)) {
    g_init_error = err;
    g_context = nullptr;
  }
}

static bool ensure_cuda(std::string &err) {
  static std::once_flag flag;
  std::call_once(flag, cuda_init_once);
  if (!g_init_error.empty()) { err = g_init_error; return false; }
  if (!g_context) { err = "CUDA context not initialised"; return false; }
  // The primary context must be made current for the *calling* thread.
  CUresult rc = cuCtxSetCurrent(g_context);
  if (!cu_check(rc, "cuCtxSetCurrent", err)) return false;
  return true;
}

// ── Grid / block helper ───────────────────────────────────────────────────────

static void launch_dims(uint64_t n_combos,
                        unsigned int &grid_x, unsigned int &block_x) {
  block_x = static_cast<unsigned int>(std::min<uint64_t>(256, n_combos));
  if (block_x == 0) block_x = 1;
  uint64_t g = (n_combos + block_x - 1) / block_x;
  grid_x = static_cast<unsigned int>(std::min<uint64_t>(65535, g));
  if (grid_x == 0) grid_x = 1;
}

// ── Device capability query ───────────────────────────────────────────────────

extern "C" int
cast_cuda_device_sm(uint32_t *out_major, uint32_t *out_minor,
                    char *err_buf, size_t err_buf_len) {
  std::string err;
  if (!ensure_cuda(err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }
  int major = 0, minor = 0;
  CUresult rc = cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, g_device);
  if (!cu_check(rc, "cuDeviceGetAttribute (major)", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }
  rc = cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, g_device);
  if (!cu_check(rc, "cuDeviceGetAttribute (minor)", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }
  if (out_major) *out_major = static_cast<uint32_t>(major);
  if (out_minor) *out_minor = static_cast<uint32_t>(minor);
  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

// ── Stateless CUDA module loading ─────────────────────────────────────────────

extern "C" void *
cast_cuda_ptx_load(const char *ptx_data, char *err_buf, size_t err_buf_len) {
  if (!ptx_data) {
    write_error_message(err_buf, err_buf_len, "ptx_data must not be null");
    return nullptr;
  }
  std::string err;
  if (!ensure_cuda(err)) {
    write_error_message(err_buf, err_buf_len, err);
    return nullptr;
  }
  CUmodule mod = nullptr;
  CUresult rc = cuModuleLoadData(&mod, ptx_data);
  if (!cu_check(rc, "cuModuleLoadData", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return nullptr;
  }
  clear_error_buffer(err_buf, err_buf_len);
  return static_cast<void *>(mod);
}

extern "C" void
cast_cuda_module_unload(void *cu_module) {
  if (cu_module) cuModuleUnload(static_cast<CUmodule>(cu_module));
}

extern "C" void *
cast_cuda_module_get_function(void *cu_module, const char *func_name,
                              char *err_buf, size_t err_buf_len) {
  if (!cu_module || !func_name) {
    write_error_message(err_buf, err_buf_len,
                        "cu_module and func_name must not be null");
    return nullptr;
  }
  CUfunction func = nullptr;
  std::string err;
  CUresult rc = cuModuleGetFunction(&func, static_cast<CUmodule>(cu_module),
                                    func_name);
  if (!cu_check(rc, "cuModuleGetFunction", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return nullptr;
  }
  clear_error_buffer(err_buf, err_buf_len);
  return static_cast<void *>(func);
}

extern "C" int
cast_cuda_kernel_apply(void *cu_function,
                       uint64_t sv_dptr,
                       uint32_t n_gate_qubits, uint32_t sv_n_qubits,
                       uint8_t /*precision*/,
                       char *err_buf, size_t err_buf_len) {
  if (!cu_function) {
    write_error_message(err_buf, err_buf_len, "cu_function must not be null");
    return 1;
  }
  if (sv_n_qubits < n_gate_qubits) {
    write_error_message(err_buf, err_buf_len,
        "statevector has fewer qubits than the gate kernel requires");
    return 1;
  }

  CUdeviceptr sv_d   = static_cast<CUdeviceptr>(sv_dptr);
  CUdeviceptr mat_d  = 0;
  uint64_t    n_combos = UINT64_C(1) << (sv_n_qubits - n_gate_qubits);

  void *args[] = {&sv_d, &mat_d, &n_combos};

  unsigned int grid_x, block_x;
  launch_dims(n_combos, grid_x, block_x);

  std::string err;
  CUresult rc = cuLaunchKernel(static_cast<CUfunction>(cu_function),
                               grid_x, 1, 1,
                               block_x, 1, 1,
                               /*shared_bytes=*/0,
                               /*stream=*/nullptr,
                               args, nullptr);
  if (!cu_check(rc, "cuLaunchKernel", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }

  rc = cuCtxSynchronize();
  if (!cu_check(rc, "cuCtxSynchronize", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }

  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

// ── Stateless device memory ───────────────────────────────────────────────────

extern "C" uint64_t
cast_cuda_sv_alloc(size_t n_elements, uint8_t precision,
                   char *err_buf, size_t err_buf_len) {
  std::string err;
  if (!ensure_cuda(err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 0;
  }
  size_t scalar_bytes = (precision == 0) ? sizeof(float) : sizeof(double);
  CUdeviceptr dptr = 0;
  CUresult rc = cuMemAlloc(&dptr, n_elements * scalar_bytes);
  if (!cu_check(rc, "cuMemAlloc", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 0;
  }
  clear_error_buffer(err_buf, err_buf_len);
  return static_cast<uint64_t>(dptr);
}

extern "C" void
cast_cuda_sv_free(uint64_t dptr) {
  if (dptr) cuMemFree(static_cast<CUdeviceptr>(dptr));
}

extern "C" int
cast_cuda_sv_zero(uint64_t dptr, size_t n_elements, uint8_t precision,
                  char *err_buf, size_t err_buf_len) {
  CUdeviceptr d = static_cast<CUdeviceptr>(dptr);
  size_t scalar_bytes = (precision == 0) ? sizeof(float) : sizeof(double);

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

extern "C" int
cast_cuda_sv_upload(uint64_t dptr, const double *host_data, size_t n_elements,
                    uint8_t precision, char *err_buf, size_t err_buf_len) {
  CUdeviceptr d = static_cast<CUdeviceptr>(dptr);
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

extern "C" int
cast_cuda_sv_download(uint64_t dptr, double *host_data, size_t n_elements,
                      uint8_t precision, char *err_buf, size_t err_buf_len) {
  CUdeviceptr d = static_cast<CUdeviceptr>(dptr);
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
