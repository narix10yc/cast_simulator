#include "simulation/KernelManager.h"
#include "simulation/KernelGenInternal.h"

#include "llvm/IR/IntrinsicsNVPTX.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/TargetRegistry.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/IR/Verifier.h"

#include "cast/LegacyQuantumGate.h"
#include "cast/CircuitGraph.h"

#include "utils/TaskDispatcher.h"
#include "utils/iocolor.h"
#include "utils/utils.h"
#include "utils/Formats.h"

#define DEBUG_TYPE "kernel-mgr-cuda"
#include <llvm/Support/Debug.h>
// #define LLVM_DEBUG(X) X

using namespace cast;
using namespace llvm;

std::ostream& CUDAKernelGenConfig::displayInfo(std::ostream& os) const {
  os << std::scientific;
  os << CYAN("=== GPU Kernel Gen Config ===\n")
     << "precision:  " << precision << "\n";

  os << "forceDenseKernel : " << forceDenseKernel << "\n"
     << "zeroTolerance : " << zeroTol << "\n"
     << "oneTolerance : " << oneTol << "\n"
     << "matrixLoadMode: ";
  switch (this->matrixLoadMode) {
    case UseMatImmValues:
      os << "UseMatImmValues\n"; break;
    case LoadInDefaultMemSpace:
      os << "LoadInDefaultMemSpace\n"; break;
    case LoadInConstMemSpace:
      os << "LoadInConstMemSpace\n"; break;
  }

  os << CYAN("================================\n");
  return os;
}

void CUDAKernelManager::emitPTX(
    int nThreads, OptimizationLevel optLevel, int verbose) {
  assert(nThreads > 0);

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  applyLLVMOptimization(nThreads, optLevel, /* progressBar */ verbose > 0);

  // Check registry info to make sure LLVM is built for NVPTX
  std::string targetTriple = "nvptx64-nvidia-cuda";
  std::string err;
  const auto* target = TargetRegistry::lookupTarget(targetTriple, err);
  if (!target) {
    errs() << RED("[Error]: ") << "Failed to lookup target. Error trace: "
           << err << "\n";
    return;
  }

  #ifdef CAST_USE_CUDA
  // Query device for compute capability
  cuInit(0);
  CUdevice device;
  cuDeviceGet(&device, 0);

  int major = 0, minor = 0;
  cuDeviceGetAttribute(
    &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  cuDeviceGetAttribute(
    &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);

  std::ostringstream archOss;
  archOss << "sm_" << major << minor;
  std::string archString = archOss.str();
  #else
  // Fallback to default compute capability
  std::string archString = "sm_76";
  #endif // #ifdef CAST_USE_CUDA

  const auto createTargetMachine = [&]() -> TargetMachine* {
    return target->createTargetMachine(
      targetTriple,   // target triple
      archString,     // cpu
      "",             // features
      {},             // options
      std::nullopt    // RM
    );
  };

  for (auto& [ctx, mod] : llvmContextModulePairs) {
    mod->setTargetTriple(targetTriple);
    mod->setDataLayout(createTargetMachine()->createDataLayout());
    if (llvm::verifyModule(*mod, &llvm::errs())) {
      llvm::errs() << "Module verification failed!\n";
      return;
    }
  }

  assert(_cudaKernels.size() == llvmContextModulePairs.size());
  utils::TaskDispatcher dispatcher(nThreads);

  for (unsigned i = 0; i < _cudaKernels.size(); i++) {
    dispatcher.enqueue([&, i=i]() {
      raw_svector_ostream sstream(_cudaKernels[i].ptxString);
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

std::vector<CUDAKernelInfo*>
CUDAKernelManager::collectCUDAKernelsFromCircuitGraph(
    const std::string& graphName) {
  std::vector<CUDAKernelInfo*> kernelInfos;
  const auto mangledName = internal::mangleGraphName(graphName);
  for (auto& kernel : _cudaKernels) {
    if (kernel.llvmFuncName.starts_with(mangledName))
      kernelInfos.push_back(&kernel);
  }
  return kernelInfos;
}

#ifdef CAST_USE_CUDA

#include "utils/cuda_api_call.h"

static int getFirstVisibleDevice() {
  const char* castEnvVar = std::getenv("CAST_USE_CUDA_DEVICE");
  if (castEnvVar != nullptr)
    return std::stoi(castEnvVar);

  const char* cudaEnvVar = std::getenv("CUDA_VISIBLE_DEVICES");
  if (cudaEnvVar != nullptr)
    return std::stoi(cudaEnvVar);

  // If neither CAST_USE_CUDA_DEVICE nor CUDA_VISIBLE_DEVICES is set, return 0
  return 0;
}

void CUDAKernelManager::initCUJIT(int nThreads, int verbose) {
  assert(jitState == JIT_PTXEmitted);
  assert(nThreads > 0);
  auto nKernels = _cudaKernels.size();
  if (nKernels < nThreads) {
    std::cerr << YELLOW("[Warning] ")
      << "Calling initCUJIT with "
      << nThreads << " threads when there are only "
      << nKernels << " kernels. Set nThreads to "
      << nKernels << " instead.\n";
    nThreads = nKernels;
  }

  cuInit(0);
  CUdevice cuDevice;
  // int deviceIdx = getFirstVisibleDevice();
  int deviceIdx = 0;
  /* It seems cuDeviceGet expects logical index.
   * So if CUDA_VISIBLE_DEVICES="2,3", cuDeviceGet(&cuDevice, 0) selects
   * physical device 2.
   * Therefore, we always choose deviceIdx to be 0, and ask users to control
   * via environment variable CUDA_VISIBLE_DEVICES
   */
  CU_CALL(cuDeviceGet(&cuDevice, deviceIdx), "Get CUDA device");
  
  // Create CUDA contexts
  cuContexts.resize(nThreads);
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
    auto& kernel = _cudaKernels[i];
    std::string ptxString(kernel.ptxString.str());
    CUcontext* cuContextPtr = &(kernel.cuTuple.cuContext);
    CUmodule* cuModulePtr = &(kernel.cuTuple.cuModule);
    CUfunction* cuFunctionPtr = &(kernel.cuTuple.cuFunction);
    const char* funcName = kernel.llvmFuncName.c_str();
    dispatcher.enqueue([=, &sharedMemValues, this, &dispatcher]() {
      auto workerID = dispatcher.getWorkerID();

      CU_CALL(cuCtxSetCurrent(cuContexts[workerID]), "cuCtxSetCurrent");
      *cuContextPtr = cuContexts[workerID];
      CU_CALL(
        cuModuleLoadData(cuModulePtr, ptxString.c_str()), "cuModuleLoadData");
      CU_CALL(
        cuModuleGetFunction(cuFunctionPtr, *cuModulePtr, funcName),
        "cuModuleGetFunction");
      int staticShared = 0;
      CU_CALL(cuFuncGetAttribute(
          &staticShared, 
          CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
          *cuFunctionPtr // kernel.cuTuple.cuFunction
      ), "cuFuncGetAttribute(SHARED_SIZE_BYTES)");
      sharedMemValues[i] = static_cast<size_t>(staticShared);
    });
  }
  if (verbose > 0)
    std::cerr << "Loading PTX codes and getting CUDA functions...\n";
  dispatcher.sync(/* progressBar */ verbose > 0);
  for (unsigned i = 0; i < nKernels; ++i) {
    _cudaKernels[i].cuTuple.sharedMemBytes = sharedMemValues[i];
  }

  jitState = JIT_CUFunctionLoaded;
}

void CUDAKernelManager::launchCUDAKernel(
    void* dData, int nQubits, CUDAKernelInfo& kernelInfo, int blockSize) {
  assert(dData != nullptr);
  assert(kernelInfo.cuTuple.cuContext != nullptr);
  assert(kernelInfo.cuTuple.cuModule != nullptr);
  assert(kernelInfo.cuTuple.cuFunction != nullptr);
  assert(blockSize == 32 || blockSize == 64 || blockSize == 128 ||
         blockSize == 256 || blockSize == 512);
  cuCtxSetCurrent(kernelInfo.cuTuple.cuContext);

  // minimum value = 5, corresponding to blockSize = 32 (warp size)
  int blockSizeInNumBits;
  if (blockSize >= 512) blockSizeInNumBits = 9;
  else if (blockSize >= 256) blockSizeInNumBits = 8;
  else if (blockSize >= 128) blockSizeInNumBits = 7;
  else if (blockSize >= 64) blockSizeInNumBits = 6;
  else blockSizeInNumBits = 5;

  blockSize = 1 << blockSizeInNumBits;

  int nGateQubits = kernelInfo.gate->nQubits();
  int gridSizeInNumBits = nQubits - blockSizeInNumBits - nGateQubits;
  assert(gridSizeInNumBits >= 0 && "gridSize must be positive");
  int gridSize = 1 << (nQubits - blockSizeInNumBits - nGateQubits);

  void* cMatPtr = kernelInfo.gate->gateMatrix.getConstantMatrix()->data();
  void* kernelParams[] = { &dData, &cMatPtr };

  CU_CALL(
    cuLaunchKernel(
      kernelInfo.cuTuple.cuFunction, 
      gridSize, 1, 1,  // grid dim
      blockSize, 1, 1, // block dim
      kernelInfo.cuTuple.sharedMemBytes,  // shared mem size
      0,              // stream
      kernelParams,  // kernel params
      nullptr         // extra options
    ), "cuLaunchKernel");
}

void CUDAKernelManager::launchCUDAKernelParam(
    void* dData,    // pointer to device statevector
    int nQubits,
    CUDAKernelInfo& kernelInfo,
    void* dMatPtr,  // pointer to device matrix
    int blockSize
)
{
  assert(dData != nullptr);
  assert(dMatPtr != nullptr);
  assert(kernelInfo.cuTuple.cuContext != nullptr);
  assert(kernelInfo.cuTuple.cuModule != nullptr);
  assert(kernelInfo.cuTuple.cuFunction != nullptr);
  assert(blockSize == 32 || blockSize == 64 || blockSize == 128 ||
         blockSize == 256 || blockSize == 512);
  cuCtxSetCurrent(kernelInfo.cuTuple.cuContext);

  // minimum value = 5, corresponding to blockSize = 32 (warp size)
  int blockSizeInNumBits;
  if (blockSize >= 512) blockSizeInNumBits = 9;
  else if (blockSize >= 256) blockSizeInNumBits = 8;
  else if (blockSize >= 128) blockSizeInNumBits = 7;
  else if (blockSize >= 64)  blockSizeInNumBits = 6;
  else blockSizeInNumBits = 5;

  int nGateQubits = kernelInfo.gate->nQubits();
  int gridSizeInNumBits = nQubits - blockSizeInNumBits - nGateQubits;
  assert(gridSizeInNumBits >= 0 && "gridSize must be positive");
  int gridSize = 1 << gridSizeInNumBits;

  void* kernelParams[] = { &dData, &dMatPtr };

  size_t matrixSize = (1 << (2*kernelInfo.gate->nQubits())) * 
                       (kernelInfo.precision == 32 ? 8 : 16);
  CU_CALL(
    cuLaunchKernel(
      kernelInfo.cuTuple.cuFunction,
      gridSize, 1, 1,    // grid dims
      blockSize, 1, 1,   // block dims
      kernelInfo.cuTuple.sharedMemBytes + matrixSize,  // sharedMem
      0,                 // stream
      kernelParams,      // kernel args
      nullptr            // extra
    ),
    "launchCUDAKernelParam"
  );
}

#endif // CAST_USE_CUDA

#undef DEBUG_TYPE