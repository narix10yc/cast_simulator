#include "simulation/KernelManager.h"
#include "simulation/KernelGenInternal.h"

#include "llvm/IR/IntrinsicsNVPTX.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/TargetRegistry.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/IR/Verifier.h"

#include "cast/QuantumGate.h"
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

  const auto createTargetMachine = [&]() -> TargetMachine* {
    return target->createTargetMachine(
      targetTriple,   // target triple
      "sm_70",        // cpu
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
  int deviceIdx = getFirstVisibleDevice();
  auto cuResult = cuDeviceGet(&cuDevice, deviceIdx);
  if (cuResult != CUDA_SUCCESS) {
    std::cerr << RED("[CUDA Err] ") << "Failed to get CUDA device "
              << deviceIdx << ". Error code " << cuResult << "\n";
  }

  utils::TaskDispatcher dispatcher(nThreads);

  // Create CUDA contexts. Each thread creates and manages its own CUDA context.
  cuContexts.resize(nThreads);
  for (unsigned t = 0; t < nThreads; ++t) {
    dispatcher.enqueue([&]() {
      auto workerID = dispatcher.getWorkerID();
      CUcontext* cuContextPtr = &cuContexts[workerID];
      CUresult cuResult = cuCtxCreate(cuContextPtr, 0, cuDevice);
      if (cuResult != CUDA_SUCCESS) {
        std::cerr << RED("[CUDA Err] ") << "Worker " << workerID
                  << " failed to create CUDA context. Error code " 
                  << cuResult << "\n";
        return;
      } else {
        LLVM_DEBUG(
          std::cerr << GREEN("[CUDA] ") << "Worker " << workerID
                    << " created CUcontext " << *cuContextPtr
                    << " at " << cuContextPtr << "\n";
        );
      }
    });
  }
  dispatcher.sync();

  // Load PTX codes
  assert(nKernels == llvmContextModulePairs.size());
  
  // TODO: Currently ptxString is captured by value. This seems to be due to the
  // property of llvm::SmallVector<char, 0> -- calling str() returns an empty
  // StringRef.
  // One fix is to replace PTXStringType from SmallVector<char, 0> to 
  // std::string. Then we need to adjust emitPTX accordingly.
  for (unsigned i = 0; i < nKernels; ++i) {
    auto& kernel = _cudaKernels[i];
    std::string ptxString(kernel.ptxString.str());
    CUcontext* cuContextPtr = &(kernel.cuTuple.cuContext);
    CUmodule* cuModulePtr = &(kernel.cuTuple.cuModule);
    CUfunction* cuFunctionPtr = &(kernel.cuTuple.cuFunction);
    const char* funcName = kernel.llvmFuncName.c_str();
    dispatcher.enqueue([=, this, &dispatcher]() {
      auto workerID = dispatcher.getWorkerID();
      CUresult cuResult;

      cuResult = cuCtxSetCurrent(cuContexts[workerID]);
      *cuContextPtr = cuContexts[workerID];
      if (cuResult != CUDA_SUCCESS) {
        std::cerr << RED("[CUDA Err] ") << "Worker " << workerID
                  << " failed to set CUDA context. Error code " 
                  << cuResult << "\n";
        return;
      }
      cuResult = cuModuleLoadData(cuModulePtr, ptxString.c_str());
      if (cuResult != CUDA_SUCCESS) {
        std::cerr << RED("[CUDA Err] ") << "Worker " << workerID
                << " failed to create CUDA module " << i << ". Error code "
                << cuResult << "\n";
        return;
      } else {
        LLVM_DEBUG(
          std::cerr << GREEN("[CUDA] ") << "Worker " << workerID
                    << " created CUmodule " << *cuModulePtr
                    << " at " << cuModulePtr << "\n";
        );
      }

      cuResult = cuModuleGetFunction(cuFunctionPtr, *cuModulePtr, funcName);
      if (cuResult != CUDA_SUCCESS) {
        std::cerr << RED("[CUDA Err] ") << "Worker " << workerID
                  << " failed to load CUDA function " << i << ". Error code "
                  << cuResult << "\n";
        return;
      } else {
        LLVM_DEBUG(
          std::cerr << GREEN("[CUDA] ") << "Worker " << workerID
                    << " loaded CUfunction " << *cuFunctionPtr
                    << " at " << cuFunctionPtr << "\n";
        );
      }
    });
  }
  if (verbose > 0)
    std::cerr << "Loading PTX codes and getting CUDA functions...\n";
  dispatcher.sync(/* progressBar */ verbose > 0);

  jitState = JIT_CUFunctionLoaded;
}

void CUDAKernelManager::launchCUDAKernel(
    void* dData, int nQubits, CUDAKernelInfo& kernelInfo) {
  assert(dData != nullptr);
  assert(kernelInfo.cuTuple.cuContext != nullptr);
  assert(kernelInfo.cuTuple.cuModule != nullptr);
  assert(kernelInfo.cuTuple.cuFunction != nullptr);
  cuCtxSetCurrent(kernelInfo.cuTuple.cuContext);

  // This corresponds to 128 threads per block
  int blockSizeInNumBits = 5;
  int nGateQubits = kernelInfo.gate->nQubits();
  int gridSizeInNumBits = nQubits - blockSizeInNumBits - nGateQubits;
  assert(gridSizeInNumBits >= 0 && "gridSize must be positive");

  unsigned blockSize = 1 << blockSizeInNumBits;
  unsigned gridSize = 1 << (nQubits - blockSizeInNumBits - nGateQubits);

  void* cMatPtr = kernelInfo.gate->gateMatrix.getConstantMatrix()->data();
  CUdeviceptr dArrDevicePtr = (CUdeviceptr)dData;
  void* kernelParams[] = { &dData, &cMatPtr };

  auto cuResult = cuLaunchKernel(
    kernelInfo.cuTuple.cuFunction, 
    gridSize, 1, 1,  // grid dim
    blockSize, 1, 1, // block dim
    0,              // shared mem size
    0,              // stream
    kernelParams,  // kernel params
    nullptr         // extra options
  );

  if (cuResult != CUDA_SUCCESS) {
    std::cerr << RED("[CUDA Driver Error]:") << "Failed to launch kernel"
              << "<<<" << gridSize << ", " << blockSize << ">>>. Error code: "
              << cuResult << "\n"; 
  }
}


#endif // CAST_USE_CUDA

#undef DEBUG_TYPE