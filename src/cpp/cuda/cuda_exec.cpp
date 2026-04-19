#include "../include/ffi_cuda.h"
#include "cuda_util.hpp"

#include <cuda.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// ── CUDA driver error helpers ─────────────────────────────────────────────────

static std::string cuErrorString(CUresult rc) {
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

static bool cuCheck(CUresult rc, const char *op, std::string &errOut) {
  if (rc == CUDA_SUCCESS)
    return true;
  errOut = std::string(op) + " failed: " + cuErrorString(rc);
  return false;
}

// ── Per-process CUDA initialisation ──────────────────────────────────────────

static CUdevice gDevice = 0;
static CUcontext gContext = nullptr;
static std::string gInitError;

static void cudaInitOnce() {
  std::string err;
  CUresult rc = cuInit(0);
  if (!cuCheck(rc, "cuInit", err)) {
    gInitError = err;
    return;
  }

  rc = cuDeviceGet(&gDevice, 0);
  if (!cuCheck(rc, "cuDeviceGet", err)) {
    gInitError = err;
    return;
  }

  rc = cuDevicePrimaryCtxRetain(&gContext, gDevice);
  if (!cuCheck(rc, "cuDevicePrimaryCtxRetain", err)) {
    gInitError = err;
    return;
  }

  rc = cuCtxSetCurrent(gContext);
  if (!cuCheck(rc, "cuCtxSetCurrent", err)) {
    gInitError = err;
    gContext = nullptr;
  }
}

static bool ensureCuda(std::string &err) {
  static std::once_flag flag;
  std::call_once(flag, cudaInitOnce);
  if (!gInitError.empty()) {
    err = gInitError;
    return false;
  }
  if (!gContext) {
    err = "CUDA context not initialized";
    return false;
  }
  // The primary context must be made current for the *calling* thread.
  CUresult const rc = cuCtxSetCurrent(gContext);
  return cuCheck(rc, "cuCtxSetCurrent", err);
}

// ── Grid / block helper ───────────────────────────────────────────────────────

static void launchDims(uint64_t nCombos, uint32_t nGateQubits, unsigned int &gridX,
                       unsigned int &blockX) {
  // 4q+ gates generate large straight-line kernels with high register
  // pressure.  Using block=256 forces the register allocator to spill
  // massively (295 virtual → 48 physical regs on typical 4q kernels).
  // A smaller block gives each thread more of the SM register file,
  // avoiding spills and recovering full bandwidth utilization.
  unsigned int const maxBlock = (nGateQubits >= 4) ? 64 : 256;
  blockX = static_cast<unsigned int>(std::min<uint64_t>(maxBlock, nCombos));
  if (blockX == 0)
    blockX = 1;
  uint64_t const g = (nCombos + blockX - 1) / blockX;
  gridX = static_cast<unsigned int>(std::min<uint64_t>(65535, g));
  if (gridX == 0)
    gridX = 1;
}

// ── Device capability query ───────────────────────────────────────────────────

extern "C" int cast_cuda_device_sm(uint32_t *outMajor, uint32_t *outMinor, char *errBuf,
                                   size_t errBufLen) {
  std::string err;
  if (!ensureCuda(err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }
  int major = 0;
  int minor = 0;
  CUresult rc = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, gDevice);
  if (!cuCheck(rc, "cuDeviceGetAttribute (major)", err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }
  rc = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, gDevice);
  if (!cuCheck(rc, "cuDeviceGetAttribute (minor)", err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }
  if (outMajor)
    *outMajor = static_cast<uint32_t>(major);
  if (outMinor)
    *outMinor = static_cast<uint32_t>(minor);
  clearErrBuf(errBuf, errBufLen);
  return 0;
}

extern "C" int cast_cuda_free_memory(uint64_t *outFreeBytes, uint64_t *outTotalBytes, char *errBuf,
                                     size_t errBufLen) {
  std::string err;
  if (!ensureCuda(err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  CUresult const rc = cuMemGetInfo(&freeBytes, &totalBytes);
  if (!cuCheck(rc, "cuMemGetInfo", err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }
  if (outFreeBytes)
    *outFreeBytes = static_cast<uint64_t>(freeBytes);
  if (outTotalBytes)
    *outTotalBytes = static_cast<uint64_t>(totalBytes);
  clearErrBuf(errBuf, errBufLen);
  return 0;
}

// ── PTX → cubin compilation ───────────────────────────────────────────────────

extern "C" int cast_cuda_ptx_to_cubin(const char *ptxData, uint8_t **outCubin, size_t *outCubinLen,
                                      char *errBuf, size_t errBufLen) {
  if (!ptxData || !outCubin || !outCubinLen) {
    writeErrBuf(errBuf, errBufLen, "arguments must not be null");
    return 1;
  }
  std::string err;
  if (!ensureCuda(err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }

  // Capture JIT compiler errors so PTX syntax problems surface clearly.
  char jitLog[4096] = {};
  uintptr_t const logSize = sizeof(jitLog);
  CUjit_option optKeys[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void *optVals[] = {static_cast<void *>(jitLog), reinterpret_cast<void *>(logSize)};

  CUlinkState linkState = nullptr;
  CUresult rc = cuLinkCreate(2, optKeys, optVals, &linkState);
  if (!cuCheck(rc, "cuLinkCreate", err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }

  rc = cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
                     const_cast<void *>(static_cast<const void *>(ptxData)), std::strlen(ptxData),
                     "kernel.ptx", 0, nullptr, nullptr);
  if (!cuCheck(rc, "cuLinkAddData", err)) {
    if (jitLog[0] != '\0') {
      err += " — ";
      err += jitLog;
    }
    cuLinkDestroy(linkState);
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }

  void *cubinPtr = nullptr;
  size_t cubinSize = 0;
  rc = cuLinkComplete(linkState, &cubinPtr, &cubinSize);
  if (!cuCheck(rc, "cuLinkComplete", err)) {
    if (jitLog[0] != '\0') {
      err += " — ";
      err += jitLog;
    }
    cuLinkDestroy(linkState);
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }

  // cubinPtr is owned by the linker state; copy before destroying.
  auto *buf = new uint8_t[cubinSize];
  std::memcpy(buf, cubinPtr, cubinSize);
  cuLinkDestroy(linkState);

  *outCubin = buf;
  *outCubinLen = cubinSize;
  clearErrBuf(errBuf, errBufLen);
  return 0;
}

extern "C" void cast_cuda_cubin_free(const uint8_t *cubin) { delete[] cubin; }

// ── CUDA module loading ───────────────────────────────────────────────────────

extern "C" void *cast_cuda_cubin_load(const uint8_t *cubinData, size_t /*cubinLen*/, char *errBuf,
                                      size_t errBufLen) {
  if (!cubinData) {
    writeErrBuf(errBuf, errBufLen, "cubinData must not be null");
    return nullptr;
  }
  std::string err;
  if (!ensureCuda(err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return nullptr;
  }
  CUmodule mod = nullptr;
  CUresult const rc = cuModuleLoadData(&mod, cubinData);
  if (!cuCheck(rc, "cuModuleLoadData", err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return nullptr;
  }
  clearErrBuf(errBuf, errBufLen);
  return static_cast<void *>(mod);
}

extern "C" void cast_cuda_module_unload(void *cuModule) {
  if (cuModule)
    cuModuleUnload(static_cast<CUmodule>(cuModule));
}

extern "C" void *cast_cuda_module_get_function(void *cuModule, const char *funcName, char *errBuf,
                                               size_t errBufLen) {
  if (!cuModule || !funcName) {
    writeErrBuf(errBuf, errBufLen, "cuModule and funcName must not be null");
    return nullptr;
  }
  CUfunction func = nullptr;
  std::string err;
  CUresult const rc = cuModuleGetFunction(&func, static_cast<CUmodule>(cuModule), funcName);
  if (!cuCheck(rc, "cuModuleGetFunction", err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return nullptr;
  }
  clearErrBuf(errBuf, errBufLen);
  return static_cast<void *>(func);
}

// ── CUDA stream ───────────────────────────────────────────────────────────────

extern "C" void *cast_cuda_stream_create(char *errBuf, size_t errBufLen) {
  std::string err;
  if (!ensureCuda(err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return nullptr;
  }
  CUstream stream = nullptr;
  CUresult const rc = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
  if (!cuCheck(rc, "cuStreamCreate", err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return nullptr;
  }
  clearErrBuf(errBuf, errBufLen);
  return static_cast<void *>(stream);
}

extern "C" void cast_cuda_stream_destroy(void *stream) {
  if (stream)
    cuStreamDestroy(static_cast<CUstream>(stream));
}

extern "C" int cast_cuda_stream_sync(void *stream, char *errBuf, size_t errBufLen) {
  std::string err;
  CUresult const rc = cuStreamSynchronize(static_cast<CUstream>(stream));
  if (!cuCheck(rc, "cuStreamSynchronize", err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }
  clearErrBuf(errBuf, errBufLen);
  return 0;
}

// ── Kernel launch ─────────────────────────────────────────────────────────────

extern "C" int cast_cuda_kernel_launch(void *cuFunction, void *stream, uint64_t svDptr,
                                       uint32_t nGateQubits, uint32_t svNQubits, char *errBuf,
                                       size_t errBufLen) {
  if (!cuFunction) {
    writeErrBuf(errBuf, errBufLen, "cuFunction must not be null");
    return 1;
  }
  if (svNQubits < nGateQubits) {
    writeErrBuf(errBuf, errBufLen, "statevector has fewer qubits than the gate kernel requires");
    return 1;
  }

  auto sv_d = static_cast<CUdeviceptr>(svDptr);
  CUdeviceptr mat_d = 0;
  uint64_t nCombos = UINT64_C(1) << (svNQubits - nGateQubits);

  void *args[] = {&sv_d, &mat_d, &nCombos};

  unsigned int gridX;
  unsigned int blockX;
  launchDims(nCombos, nGateQubits, gridX, blockX);

  std::string err;
  CUresult const rc =
      cuLaunchKernel(static_cast<CUfunction>(cuFunction), gridX, 1, 1, blockX, 1, 1,
                     /*sharedMemBytes=*/0, static_cast<CUstream>(stream), args, nullptr);
  if (!cuCheck(rc, "cuLaunchKernel", err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }

  clearErrBuf(errBuf, errBufLen);
  return 0;
}

// ── CUDA timing events ────────────────────────────────────────────────────────

extern "C" void *cast_cuda_event_create(char *errBuf, size_t errBufLen) {
  std::string err;
  if (!ensureCuda(err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return nullptr;
  }
  CUevent ev = nullptr;
  CUresult const rc = cuEventCreate(&ev, CU_EVENT_DEFAULT);
  if (!cuCheck(rc, "cuEventCreate", err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return nullptr;
  }
  clearErrBuf(errBuf, errBufLen);
  return static_cast<void *>(ev);
}

extern "C" int cast_cuda_event_record(void *event, void *stream, char *errBuf, size_t errBufLen) {
  if (!event || !stream) {
    writeErrBuf(errBuf, errBufLen, "event and stream must not be null");
    return 1;
  }
  std::string err;
  CUresult const rc = cuEventRecord(static_cast<CUevent>(event), static_cast<CUstream>(stream));
  if (!cuCheck(rc, "cuEventRecord", err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }
  clearErrBuf(errBuf, errBufLen);
  return 0;
}

extern "C" int cast_cuda_event_synchronize(void *event, char *errBuf, size_t errBufLen) {
  if (!event) {
    writeErrBuf(errBuf, errBufLen, "event must not be null");
    return 1;
  }
  std::string err;
  CUresult const rc = cuEventSynchronize(static_cast<CUevent>(event));
  if (!cuCheck(rc, "cuEventSynchronize", err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }
  clearErrBuf(errBuf, errBufLen);
  return 0;
}

extern "C" void cast_cuda_event_destroy(void *event) {
  if (event)
    cuEventDestroy(static_cast<CUevent>(event));
}

extern "C" int cast_cuda_event_elapsed_ms(void *startEvent, void *endEvent, float *outMs,
                                          char *errBuf, size_t errBufLen) {
  if (!startEvent || !endEvent || !outMs) {
    writeErrBuf(errBuf, errBufLen, "arguments must not be null");
    return 1;
  }
  std::string err;
  CUresult const rc =
      cuEventElapsedTime(outMs, static_cast<CUevent>(startEvent), static_cast<CUevent>(endEvent));
  if (!cuCheck(rc, "cuEventElapsedTime", err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }
  clearErrBuf(errBuf, errBufLen);
  return 0;
}

// ── Device-to-device async memcpy ─────────────────────────────────────────────

extern "C" int cast_cuda_memcpy_dtod_async(uint64_t dst, uint64_t src, size_t nBytes, void *stream,
                                           char *errBuf, size_t errBufLen) {
  std::string err;
  if (!ensureCuda(err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }
  CUresult const rc =
      cuMemcpyDtoDAsync(static_cast<CUdeviceptr>(dst), static_cast<CUdeviceptr>(src), nBytes,
                        static_cast<CUstream>(stream));
  if (!cuCheck(rc, "cuMemcpyDtoDAsync", err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }
  clearErrBuf(errBuf, errBufLen);
  return 0;
}

// ── Stateless device memory ───────────────────────────────────────────────────

extern "C" uint64_t cast_cuda_sv_alloc(size_t nElements, uint8_t precision, char *errBuf,
                                       size_t errBufLen) {
  std::string err;
  if (!ensureCuda(err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 0;
  }
  size_t const scalarBytes = (precision == 0) ? sizeof(float) : sizeof(double);
  CUdeviceptr dptr = 0;
  CUresult const rc = cuMemAlloc(&dptr, nElements * scalarBytes);
  if (!cuCheck(rc, "cuMemAlloc", err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 0;
  }
  clearErrBuf(errBuf, errBufLen);
  return static_cast<uint64_t>(dptr);
}

extern "C" void cast_cuda_sv_free(uint64_t dptr) {
  if (dptr)
    cuMemFree(static_cast<CUdeviceptr>(dptr));
}

extern "C" int cast_cuda_sv_zero(uint64_t dptr, size_t nElements, uint8_t precision, char *errBuf,
                                 size_t errBufLen) {
  auto const d = static_cast<CUdeviceptr>(dptr);
  size_t const scalarBytes = (precision == 0) ? sizeof(float) : sizeof(double);

  std::string err;
  CUresult rc = cuMemsetD8(d, 0, nElements * scalarBytes);
  if (!cuCheck(rc, "cuMemsetD8", err)) {
    writeErrBuf(errBuf, errBufLen, err);
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
  if (!cuCheck(rc, "cuMemcpyHtoD (zero state)", err)) {
    writeErrBuf(errBuf, errBufLen, err);
    return 1;
  }

  clearErrBuf(errBuf, errBufLen);
  return 0;
}

extern "C" int cast_cuda_sv_upload(uint64_t dptr, const double *hostData, size_t nElements,
                                   uint8_t precision, char *errBuf, size_t errBufLen) {
  auto const d = static_cast<CUdeviceptr>(dptr);
  std::string err;
  CUresult rc;

  if (precision == 1) { // F64: direct copy
    rc = cuMemcpyHtoD(d, hostData, nElements * sizeof(double));
    if (!cuCheck(rc, "cuMemcpyHtoD", err)) {
      writeErrBuf(errBuf, errBufLen, err);
      return 1;
    }
  } else { // F32: narrow on host then upload
    std::vector<float> fdata(nElements);
    for (size_t i = 0; i < nElements; ++i)
      fdata[i] = static_cast<float>(hostData[i]);
    rc = cuMemcpyHtoD(d, fdata.data(), nElements * sizeof(float));
    if (!cuCheck(rc, "cuMemcpyHtoD (f32)", err)) {
      writeErrBuf(errBuf, errBufLen, err);
      return 1;
    }
  }

  clearErrBuf(errBuf, errBufLen);
  return 0;
}

extern "C" int cast_cuda_sv_download(uint64_t dptr, double *hostData, size_t nElements,
                                     uint8_t precision, char *errBuf, size_t errBufLen) {
  auto const d = static_cast<CUdeviceptr>(dptr);
  std::string err;
  CUresult rc;

  if (precision == 1) { // F64: direct copy
    rc = cuMemcpyDtoH(hostData, d, nElements * sizeof(double));
    if (!cuCheck(rc, "cuMemcpyDtoH", err)) {
      writeErrBuf(errBuf, errBufLen, err);
      return 1;
    }
  } else { // F32: download then widen to f64
    std::vector<float> fdata(nElements);
    rc = cuMemcpyDtoH(fdata.data(), d, nElements * sizeof(float));
    if (!cuCheck(rc, "cuMemcpyDtoH (f32)", err)) {
      writeErrBuf(errBuf, errBufLen, err);
      return 1;
    }
    for (size_t i = 0; i < nElements; ++i)
      hostData[i] = static_cast<double>(fdata[i]);
  }

  clearErrBuf(errBuf, errBufLen);
  return 0;
}
