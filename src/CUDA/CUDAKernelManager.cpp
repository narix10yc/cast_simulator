#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/Config.h"

#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

#include "utils/Formats.h"
#include "utils/TaskDispatcher.h"
#include "cast/CUDA/Config.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

#define DEBUG_TYPE "kernel-mgr-cuda"
#include "llvm/Support/Debug.h"
// #define LLVM_DEBUG(X) X

using namespace cast;
using namespace llvm;

std::ostream& CUDAKernelGenConfig::displayInfo(std::ostream& os) const {
  os << std::scientific;
  os << CYAN("=== GPU Kernel Gen Config ===\n") << "precision: f"
     << static_cast<int>(precision) << "\n"
     << "forceDenseKernel : " << forceDenseKernel << "\n"
     << "zeroTolerance : " << zeroTol << "\n"
     << "oneTolerance : " << oneTol << "\n"
     << "matrixLoadMode: ";
  switch (this->matrixLoadMode) {
  case CUDAMatrixLoadMode::UseMatImmValues:
    os << "UseMatImmValues\n";
    break;
  case CUDAMatrixLoadMode::LoadInDefaultMemSpace:
    os << "LoadInDefaultMemSpace\n";
    break;
  case CUDAMatrixLoadMode::LoadInConstMemSpace:
    os << "LoadInConstMemSpace\n";
    break;
  }

  os << CYAN("================================\n");
  return os;
}

void CUDAKernelManager::emitPTX(int nThreads,
                                OptimizationLevel optLevel,
                                int verbose) {
  assert(nThreads > 0);

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  applyLLVMOptimization(nThreads, optLevel, /* progressBar */ verbose > 0);

  std::string targetTriple = "nvptx64-nvidia-cuda";
  std::string err;
  const auto* target = TargetRegistry::lookupTarget(targetTriple, err);
  if (!target) {
    errs() << RED("[Error]: ")
           << "Failed to lookup target. Error trace: " << err << "\n";
    return;
  }

#ifdef CAST_USE_CUDA
  int major = 0, minor = 0;
  cast::getCudaComputeCapability(major, minor);
  std::ostringstream archOss;
  archOss << "sm_" << major << minor;
  std::string archString = archOss.str();
#else
  // For non-CUDA builds, set default architecture to sm_76
  std::string archString = "sm_76";
#endif

  const auto createTargetMachine = [&]() -> TargetMachine* {
    return target->createTargetMachine(
        targetTriple, archString, "", {}, std::nullopt);
  };

  for (auto& [ctx, mod] : llvmContextModulePairs) {
    mod->setTargetTriple(targetTriple);
    mod->setDataLayout(createTargetMachine()->createDataLayout());
    if (llvm::verifyModule(*mod, &llvm::errs())) {
      llvm::errs() << "Module verification failed, attempting to proceed with "
                      "PTX emission\n";
    }
  }

  assert(standaloneKernels_.size() == llvmContextModulePairs.size());
  utils::TaskDispatcher dispatcher(nThreads);

  for (unsigned i = 0; i < standaloneKernels_.size(); i++) {
    dispatcher.enqueue([&, i = i]() {
      raw_svector_ostream sstream(standaloneKernels_[i]->ptxString);
      legacy::PassManager passManager;
      if (createTargetMachine()->addPassesToEmitFile(
              passManager, sstream, nullptr, CodeGenFileType::AssemblyFile)) {
        errs() << "The target machine can't emit a file of this type\n";
        return;
      }
      passManager.run(*llvmContextModulePairs[i].llvmModule);
    });
  }
  if (verbose > 0)
    std::cerr << "Emitting PTX codes...\n";
  dispatcher.sync(/* progressBar */ verbose > 0);
  jitState = JIT_PTXEmitted;
}

void CUDAKernelManager::initCUJIT(int nThreads, int verbose) {
  assert(jitState == JIT_PTXEmitted);
  assert(nThreads > 0);
  auto nKernels = standaloneKernels_.size();
  if (nKernels < nThreads) {
    std::cerr << YELLOW("[Warning] ") << "Calling initCUJIT with " << nThreads
              << " threads when there are only " << nKernels
              << " kernels. Set nThreads to " << nKernels << " instead.\n";
    nThreads = nKernels;
  }

  cuInit(0);
  CUdevice cuDevice;
  // int deviceIdx = getFirstVisibleDevice();
  int deviceIdx = 0;
  /* cuDeviceGet expects logical index.
   * So if CUDA_VISIBLE_DEVICES="2,3", cuDeviceGet(&cuDevice, 0) selects
   * physical device 2.
   * Therefore, we always choose deviceIdx to be 0, and ask users to control
   * via environment variable CUDA_VISIBLE_DEVICES
   */
  CU_CALL(cuDeviceGet(&cuDevice, deviceIdx), "Get CUDA device");

  // Create CUDA contexts
  cuContexts.resize(nThreads, nullptr);
  for (unsigned t = 0; t < nThreads; ++t) {
    CU_CALL(cuCtxCreate(&cuContexts[t], 0, cuDevice), "Create CUDA context");
  }

  // Load PTX codes
  assert(nKernels == llvmContextModulePairs.size());
  utils::TaskDispatcher dispatcher(nThreads);

  // TODO: Currently ptxString is captured by value. This seems to be due to the
  // property of llvm::SmallVector<char, 0> -- calling str() returns an empty
  // StringRef.
  // One fix is to replace PTXStringType from SmallVector<char, 0> to
  // std::string. Then we need to adjust emitPTX accordingly.
  std::vector<size_t> sharedMemValues(nKernels);

  for (unsigned i = 0; i < nKernels; ++i) {
    auto& kernel = standaloneKernels_[i];
    std::string ptxString(kernel->ptxString.str());
    CUcontext* cuContextPtr = &(kernel->cuTuple.cuContext);
    CUmodule* cuModulePtr = &(kernel->cuTuple.cuModule);
    CUfunction* cuFunctionPtr = &(kernel->cuTuple.cuFunction);
    const char* funcName = kernel->llvmFuncName.c_str();
    dispatcher.enqueue([=, &sharedMemValues, this, &dispatcher]() {
      auto workerID = dispatcher.getWorkerID();

      CU_CALL(cuCtxSetCurrent(cuContexts[workerID]), "cuCtxSetCurrent");
      *cuContextPtr = cuContexts[workerID];
      CU_CALL(cuModuleLoadData(cuModulePtr, ptxString.c_str()),
              "cuModuleLoadData");
      CU_CALL(cuModuleGetFunction(cuFunctionPtr, *cuModulePtr, funcName),
              "cuModuleGetFunction");
      int staticShared = 0;
      CU_CALL(cuFuncGetAttribute(&staticShared,
                                 CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                 *cuFunctionPtr // kernel->cuTuple.cuFunction
                                 ),
              "cuFuncGetAttribute(SHARED_SIZE_BYTES)");
      sharedMemValues[i] = static_cast<size_t>(staticShared);
    });
  }
  if (verbose > 0)
    std::cerr << "Loading PTX codes and getting CUDA functions...\n";
  dispatcher.sync(/* progressBar */ verbose > 0);
  for (unsigned i = 0; i < nKernels; ++i) {
    standaloneKernels_[i]->cuTuple.sharedMemBytes = sharedMemValues[i];
  }

  jitState = JIT_CUFunctionLoaded;
}

void CUDAKernelManager::launchCUDAKernel(
    void* dData, // device state‑vector
    int nQubits, // total system‑qubits
    CUDAKernelInfo& kernelInfo,
    int /* blockSize -> ignored - TILE is fixed in kernel */) {
  assert(dData != nullptr);
  assert(kernelInfo.cuTuple.cuContext && kernelInfo.cuTuple.cuFunction);

  cuCtxSetCurrent(kernelInfo.cuTuple.cuContext);

  /* parameters of the generated kernel */
  unsigned k = kernelInfo.gate->nQubits();
  unsigned N = 1u << k;
  unsigned TILE = std::min(256u, N);

  /* number of different assignments of the non‑target qubits */
  unsigned combos = 1u << (nQubits - k);         // 2^{nSys-k}
  unsigned tilesPerGate = (N + TILE - 1) / TILE; // rows per slice

  dim3 blockDim(TILE, 1, 1);
  dim3 gridDim(combos * tilesPerGate, 1, 1);

  const void* cMatPtr = nullptr;
  if (auto* stdQuGate =
          llvm::dyn_cast_or_null<StandardQuantumGate>(kernelInfo.gate.get())) {
    if (auto scalarGM = stdQuGate->getScalarGM()) {
      cMatPtr = scalarGM->matrix().data();
    }
  }

  void* kernelParams[2] = {&dData, &cMatPtr};

  // clang-format off
  CU_CALL(cuLaunchKernel(kernelInfo.cuTuple.cuFunction,
                         gridDim.x, gridDim.y, gridDim.z,
                         blockDim.x, blockDim.y, blockDim.z,
                         kernelInfo.cuTuple.sharedMemBytes,
                         /*stream*/ 0,
                         kernelParams,
                         /*extra*/ nullptr),
          "cuLaunchKernel");
  // clang-format on
}

void CUDAKernelManager::launchCUDAKernelParam(
    void* dData, // pointer to device statevector
    int nQubits,
    CUDAKernelInfo& kernelInfo,
    void* dMatPtr, // pointer to device matrix
    int blockSize  // ignored if fixed TILE is used
) {
  assert(dData != nullptr);
  assert(dMatPtr != nullptr);
  assert(kernelInfo.cuTuple.cuContext != nullptr);
  assert(kernelInfo.cuTuple.cuModule != nullptr);
  assert(kernelInfo.cuTuple.cuFunction != nullptr);
  cuCtxSetCurrent(kernelInfo.cuTuple.cuContext);

  // Define tile size to match kernel expectations
  unsigned nGateQubits = kernelInfo.gate->nQubits();
  unsigned N = 1u << nGateQubits; // Size of gate matrix (2^nGateQubits)
  unsigned TILE = std::min(256u, N);
  unsigned combos =
      (nQubits > nGateQubits) ? (1u << (nQubits - nGateQubits)) : 1;
  unsigned tilesPerGate = (N + TILE - 1) / TILE;
  unsigned gridDimX = combos * tilesPerGate;

  dim3 gridDim(gridDimX, 1, 1);
  dim3 blockDim(TILE, 1, 1);

  void* kernelParams[] = {&dData, &dMatPtr};

  // clang-format off
  CU_CALL(cuLaunchKernel(kernelInfo.cuTuple.cuFunction,
                         gridDim.x, gridDim.y, gridDim.z,
                         blockDim.x, blockDim.y, blockDim.z,
                         kernelInfo.cuTuple.sharedMemBytes, // shared memory
                         0,                                 // stream
                         kernelParams,                      // kernel arguments
                         nullptr),
          "launchCUDAKernelParam");
  // clang-format on
}

#undef DEBUG_TYPE
