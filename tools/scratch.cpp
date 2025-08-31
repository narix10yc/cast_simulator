#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"

#include "cuda.h"
#include "cuda_runtime.h"

#include "utils/utils.h"

using namespace cast;

static void myMallocDeviceData(CUDAStatevectorF64* sv) {
  CUDA_CHECK(cudaMalloc(&sv->dData_, sv->sizeInBytes()));
  // CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize device");
  // CUcontext ctx = nullptr;
  // cuCtxGetCurrent(&ctx);
  // std::cerr << "From CUDAStatevector::mallocDeviceData: current context is "
  //           << ctx << "\n";
  // CUmemorytype mt = {};
  // int devOrd = -1;
  // CUcontext ptrCtx = nullptr;
  // size_t rangeSize = 0;
  // CU_CHECK(cuPointerGetAttributes(
  //     4,
  //     (CUpointer_attribute[]){CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
  //                             CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
  //                             CU_POINTER_ATTRIBUTE_CONTEXT,
  //                             CU_POINTER_ATTRIBUTE_RANGE_SIZE},
  //     (void*[]){&mt, &devOrd, &ptrCtx, &rangeSize},
  //     reinterpret_cast<CUdeviceptr>(sv->dData_)));

  // std::cerr << "Query of " << sv->dData_ << ":\n"
  //           << " - Memory type: " << mt << "\n"
  //           << " - Device ordinal: " << devOrd << "\n"
  //           << " - Context: " << ptrCtx << "\n"
  //           << " - Range size: " << rangeSize << "\n";
  // int cDev = -1;
  // cuCtxGetDevice(&cDev);
  // int rDev = -1;
  // cudaGetDevice(&rDev);

  // CUcontext cur = nullptr;
  // cuCtxGetCurrent(&cur);
  // fprintf(stderr,
  //         "thread devs: Driver=%d Runtime=%d ctx=%p\n",
  //         cDev,
  //         rDev,
  //         (void*)cur);
}

int main(int argc, char** argv) {
  constexpr int nQubits = 28;

  CUDAStatevectorF64 sv(nQubits);
  sv.initialize();
  CUDAKernelGenConfig config;
  CUDAKernelManager km(1);

  for (int i = 0; i < 4; ++i) {
    QuantumGate::TargetQubitsType qubits;
    utils::sampleNoReplacement(nQubits, 2, qubits);
    km.genStandaloneGate(config,
                         StandardQuantumGate::RandomUnitary(qubits),
                         "gate_" + std::to_string(i))
        .consumeError();
  }

  auto r = km.initJIT(1, 1);
  if (!r) {
    std::cerr << "Failed to initialize JIT: " << r.takeError() << "\n";
    return 1;
  }

  km.setLaunchConfig(sv.getDevicePtr(), nQubits);
  for (auto& kernel : km) {
    km.enqueueKernelLaunch(kernel);
  }

  return 0;
}