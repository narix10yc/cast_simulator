#include "cuda.h"
#include "cuda_jit.h"
#include "cuda_util.h"

#include <cuda.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
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
  // The primary context must be made current for the *calling* thread; it is
  // not inherited from the initialisation thread.  cuCtxSetCurrent is cheap
  // when the context is already current.
  CUresult rc = cuCtxSetCurrent(g_context);
  if (!cu_check(rc, "cuCtxSetCurrent", err)) return false;
  return true;
}

// ── Opaque struct bodies ──────────────────────────────────────────────────────

struct ExecEntry {
  cast_cuda_kernel_id_t kernel_id    = 0;
  uint32_t              n_gate_qubits = 0;
  cast_cuda_precision_t precision    = CAST_CUDA_PRECISION_F64;
  std::string           func_name{};
  CUmodule              module   = nullptr;
  CUfunction            function = nullptr;
};

struct cast_cuda_exec_session_t {
  std::unordered_map<cast_cuda_kernel_id_t, ExecEntry> entries{};
};

struct cast_cuda_statevector_t {
  uint32_t              n_qubits   = 0;
  cast_cuda_precision_t precision  = CAST_CUDA_PRECISION_F64;
  CUdeviceptr           dptr       = 0;
  size_t                n_elements = 0; ///< 2 * 2^n_qubits
};

// ── Grid / block helper ───────────────────────────────────────────────────────

static void launch_dims(uint64_t n_combos,
                        unsigned int &grid_x, unsigned int &block_x) {
  block_x = static_cast<unsigned int>(std::min<uint64_t>(256, n_combos));
  if (block_x == 0) block_x = 1;
  uint64_t g = (n_combos + block_x - 1) / block_x;
  grid_x = static_cast<unsigned int>(std::min<uint64_t>(65535, g));
  if (grid_x == 0) grid_x = 1;
}

// ── Execution session ─────────────────────────────────────────────────────────

extern "C" cast_cuda_exec_session_t *
cast_cuda_exec_session_new(const cast_cuda_compilation_session_t *session,
                           char *err_buf, size_t err_buf_len) {
  if (!session) {
    write_error_message(err_buf, err_buf_len, "session must not be null");
    return nullptr;
  }

  std::string err;
  if (!ensure_cuda(err)) {
    write_error_message(err_buf, err_buf_len, err);
    return nullptr;
  }

  auto *exec = new cast_cuda_exec_session_t();

  for (const auto &kv : session->kernels) {
    const CastCudaCompiledKernel &compiled = kv.second;

    ExecEntry entry;
    entry.kernel_id     = compiled.kernel_id;
    entry.n_gate_qubits = compiled.n_gate_qubits;
    entry.precision     = compiled.precision;
    entry.func_name     = compiled.func_name;

    // Load PTX into a CUDA module (driver JIT-compiles it).
    CUresult rc = cuModuleLoadData(&entry.module,
                                   compiled.ptx.c_str());
    if (!cu_check(rc, "cuModuleLoadData", err)) {
      // Unload any already-loaded modules before bailing.
      for (auto &e : exec->entries)
        if (e.second.module) cuModuleUnload(e.second.module);
      delete exec;
      write_error_message(err_buf, err_buf_len,
          "failed to load PTX for kernel '" + compiled.func_name
          + "': " + err);
      return nullptr;
    }

    rc = cuModuleGetFunction(&entry.function, entry.module,
                             compiled.func_name.c_str());
    if (!cu_check(rc, "cuModuleGetFunction", err)) {
      cuModuleUnload(entry.module);
      for (auto &e : exec->entries)
        if (e.second.module) cuModuleUnload(e.second.module);
      delete exec;
      write_error_message(err_buf, err_buf_len,
          "failed to resolve function '" + compiled.func_name
          + "': " + err);
      return nullptr;
    }

    exec->entries.emplace(compiled.kernel_id, std::move(entry));
  }

  clear_error_buffer(err_buf, err_buf_len);
  return exec;
}

extern "C" void
cast_cuda_exec_session_delete(cast_cuda_exec_session_t *session) {
  if (!session) return;
  for (auto &kv : session->entries)
    if (kv.second.module) cuModuleUnload(kv.second.module);
  delete session;
}

extern "C" int
cast_cuda_exec_session_apply(cast_cuda_exec_session_t *session,
                             cast_cuda_kernel_id_t kernel_id,
                             cast_cuda_statevector_t *sv,
                             char *err_buf, size_t err_buf_len) {
  if (!session) {
    write_error_message(err_buf, err_buf_len, "exec session must not be null");
    return 1;
  }
  if (!sv) {
    write_error_message(err_buf, err_buf_len, "statevector must not be null");
    return 1;
  }

  auto it = session->entries.find(kernel_id);
  if (it == session->entries.end()) {
    write_error_message(err_buf, err_buf_len,
                        "kernel id not found in exec session");
    return 1;
  }
  const ExecEntry &entry = it->second;

  if (sv->n_qubits < entry.n_gate_qubits) {
    write_error_message(err_buf, err_buf_len,
        "statevector has fewer qubits than the gate kernel requires");
    return 1;
  }
  if (sv->precision != entry.precision) {
    write_error_message(err_buf, err_buf_len,
        "statevector precision does not match kernel precision");
    return 1;
  }

  // Kernel signature: (ptr sv, ptr mat, i64 n_combos)
  // mat is always null — matrix values are compiled as immediates.
  CUdeviceptr sv_dptr  = sv->dptr;
  CUdeviceptr mat_dptr = 0;
  uint64_t    n_combos = UINT64_C(1) << (sv->n_qubits - entry.n_gate_qubits);

  void *args[] = {&sv_dptr, &mat_dptr, &n_combos};

  unsigned int grid_x, block_x;
  launch_dims(n_combos, grid_x, block_x);

  std::string err;
  CUresult rc = cuLaunchKernel(entry.function,
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

// ── Statevector ───────────────────────────────────────────────────────────────

extern "C" cast_cuda_statevector_t *
cast_cuda_statevector_new(uint32_t n_qubits, cast_cuda_precision_t precision,
                          char *err_buf, size_t err_buf_len) {
  std::string err;
  if (!ensure_cuda(err)) {
    write_error_message(err_buf, err_buf_len, err);
    return nullptr;
  }

  size_t n_elements = static_cast<size_t>(2) << n_qubits; // 2 * 2^n_qubits
  size_t scalar_bytes =
      (precision == CAST_CUDA_PRECISION_F32) ? sizeof(float) : sizeof(double);
  size_t total_bytes = n_elements * scalar_bytes;

  CUdeviceptr dptr = 0;
  CUresult rc = cuMemAlloc(&dptr, total_bytes);
  if (!cu_check(rc, "cuMemAlloc", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return nullptr;
  }

  auto *sv      = new cast_cuda_statevector_t();
  sv->n_qubits  = n_qubits;
  sv->precision = precision;
  sv->dptr      = dptr;
  sv->n_elements = n_elements;

  clear_error_buffer(err_buf, err_buf_len);
  return sv;
}

extern "C" void
cast_cuda_statevector_delete(cast_cuda_statevector_t *sv) {
  if (!sv) return;
  if (sv->dptr) cuMemFree(sv->dptr);
  delete sv;
}

extern "C" int
cast_cuda_statevector_zero(cast_cuda_statevector_t *sv,
                           char *err_buf, size_t err_buf_len) {
  if (!sv) {
    write_error_message(err_buf, err_buf_len, "statevector must not be null");
    return 1;
  }

  size_t scalar_bytes =
      (sv->precision == CAST_CUDA_PRECISION_F32) ? sizeof(float) : sizeof(double);
  size_t total_bytes = sv->n_elements * scalar_bytes;

  std::string err;
  CUresult rc = cuMemsetD8(sv->dptr, 0, total_bytes);
  if (!cu_check(rc, "cuMemsetD8", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }

  // Write 1.0 into the real part of amplitude[0].
  if (sv->precision == CAST_CUDA_PRECISION_F32) {
    float one = 1.0f;
    rc = cuMemcpyHtoD(sv->dptr, &one, sizeof(float));
  } else {
    double one = 1.0;
    rc = cuMemcpyHtoD(sv->dptr, &one, sizeof(double));
  }
  if (!cu_check(rc, "cuMemcpyHtoD (zero state init)", err)) {
    write_error_message(err_buf, err_buf_len, err);
    return 1;
  }

  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

extern "C" int
cast_cuda_statevector_upload(cast_cuda_statevector_t *sv,
                             const double *host_data, size_t n_elements,
                             char *err_buf, size_t err_buf_len) {
  if (!sv) {
    write_error_message(err_buf, err_buf_len, "statevector must not be null");
    return 1;
  }
  if (n_elements != sv->n_elements) {
    write_error_message(err_buf, err_buf_len,
        "n_elements mismatch: expected " + std::to_string(sv->n_elements)
        + ", got " + std::to_string(n_elements));
    return 1;
  }

  std::string err;
  CUresult rc;
  if (sv->precision == CAST_CUDA_PRECISION_F64) {
    rc = cuMemcpyHtoD(sv->dptr, host_data, n_elements * sizeof(double));
    if (!cu_check(rc, "cuMemcpyHtoD", err)) {
      write_error_message(err_buf, err_buf_len, err);
      return 1;
    }
  } else {
    // Convert double[] → float[] on the host, then upload.
    std::vector<float> fdata(n_elements);
    for (size_t i = 0; i < n_elements; ++i)
      fdata[i] = static_cast<float>(host_data[i]);
    rc = cuMemcpyHtoD(sv->dptr, fdata.data(), n_elements * sizeof(float));
    if (!cu_check(rc, "cuMemcpyHtoD (f32)", err)) {
      write_error_message(err_buf, err_buf_len, err);
      return 1;
    }
  }

  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

extern "C" int
cast_cuda_statevector_download(const cast_cuda_statevector_t *sv,
                               double *host_data, size_t n_elements,
                               char *err_buf, size_t err_buf_len) {
  if (!sv) {
    write_error_message(err_buf, err_buf_len, "statevector must not be null");
    return 1;
  }
  if (n_elements != sv->n_elements) {
    write_error_message(err_buf, err_buf_len,
        "n_elements mismatch: expected " + std::to_string(sv->n_elements)
        + ", got " + std::to_string(n_elements));
    return 1;
  }

  std::string err;
  CUresult rc;
  if (sv->precision == CAST_CUDA_PRECISION_F64) {
    rc = cuMemcpyDtoH(host_data, sv->dptr, n_elements * sizeof(double));
    if (!cu_check(rc, "cuMemcpyDtoH", err)) {
      write_error_message(err_buf, err_buf_len, err);
      return 1;
    }
  } else {
    std::vector<float> fdata(n_elements);
    rc = cuMemcpyDtoH(fdata.data(), sv->dptr, n_elements * sizeof(float));
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
