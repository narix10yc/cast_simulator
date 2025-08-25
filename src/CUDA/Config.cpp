#include "cast/CUDA/Config.h"

#include "utils/iocolor.h"
#include "utils/utils.h"

#include "llvm/Config/llvm-config.h"

using namespace cast;

#include <cuda.h>

void cast::displayCUDA() {
  CU_CHECK(cuInit(0));

  int deviceCount = 0;
  CU_CHECK(cuDeviceGetCount(&deviceCount));

  if (deviceCount == 0) {
    std::cout << "No CUDA-capable devices found.\n";
    return;
  }

  std::cout << "Number of CUDA-capable GPUs: " << deviceCount << "\n";

  for (int i = 0; i < deviceCount; ++i) {
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, i));

    char name[256];
    CU_CHECK(cuDeviceGetName(name, 256, device));

    int major = 0, minor = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CU_CHECK(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    size_t totalMem = 0;
    CU_CHECK(cuDeviceTotalMem(&totalMem, device));

    int mpCount = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &mpCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));

    int clockRate = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));

    char pciBusId[32];
    CU_CHECK(cuDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device));

    std::cout << "PCI Bus ID: " << pciBusId << "\n";
    std::cout << "----- GPU " << i << " -----\n";
    std::cout << "Name: " << name << "\n";
    std::cout << "Compute Capability: " << major << "." << minor << "\n";
    std::cout << "Global Memory: " << (totalMem >> 20) << " MB\n";
    std::cout << "Multiprocessors: " << mpCount << "\n";
    std::cout << "Clock Rate: " << (clockRate / 1000) << " MHz\n";
    std::cout << "PCI Bus ID: " << pciBusId << "\n";
  }
}

void cast::getCudaComputeCapability(int& major, int& minor) {
  // The CUDA compute capability supported by LLVM NVPTX backend can be found
  // during LLVM builds here:
  // build/lib/Target/NVPTX/NVPTXGenRegisterInfo.inc
  // As of LLVM 20, the maximum supported is sm_120 and sm_120a.

#if LLVM_VERSION_MAJOR == 20
  constexpr int MAJOR_CAP = 12;
  constexpr int MINOR_CAP = 0;
#elif LLVM_VERSION_MAJOR == 19
  constexpr int MAJOR_CAP = 10;
  constexpr int MINOR_CAP = 0;
#else // For older versions, we assume sm_90 is the maximum supported.
  constexpr int MAJOR_CAP = 9;
  constexpr int MINOR_CAP = 0;
#endif

  CUresult res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    std::cerr << "cuInit failed with error code: " << res << "\n";
    return;
  }

  CUdevice device;
  CU_CHECK(cuDeviceGet(&device, 0)); // Get the first CUDA device
  CU_CHECK(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  CU_CHECK(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

  if (major < MAJOR_CAP || (major == MAJOR_CAP && minor <= MINOR_CAP))
    return;

  const char* env = std::getenv("CAST_CAP_CUDA_ARCH");
  if (env != nullptr && std::string(env) == "True") {
    major = MAJOR_CAP;
    minor = 0;
    return;
  }
  if (env != nullptr) {
    // silence the warning if CAST_CAP_CUDA_ARCH is set to other values
    return;
  }
  std::cerr << YELLOW("[Warning] ") << "CUDA compute capability " << major
            << "." << minor
            << " may not be supported by this LLVM (" LLVM_VERSION_STRING
               ") release. If you encounter "
               "issues, set environment variable CAST_CAP_CUDA_ARCH to True, "
               "and we will cap it to sm_"
            << MAJOR_CAP << MINOR_CAP
            << ". Alternatively, set "
               "CAST_CAP_CUDA_ARCH to False to silence this warning.\n";
  return;
}