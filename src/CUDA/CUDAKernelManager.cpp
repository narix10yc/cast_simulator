#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/Config.h"

#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

#include "cast/CUDA/Config.h"
#include "utils/Formats.h"
#include "utils/TaskDispatcher.h"
#include "utils/iocolor.h"
#include "utils/utils.h"
#include <fstream>

#define DEBUG_TYPE "kernel-mgr-cuda"
#include "llvm/Support/Debug.h"
// #define LLVM_DEBUG(X) X

using namespace cast;
using namespace llvm;

std::ostream& CUDAKernelGenConfig::displayInfo(std::ostream& os) const {
  os << std::scientific;
  os << CYAN("=== CUDA Kernel Gen Config ===\n") << "precision        : f"
     << static_cast<int>(precision) << "\n"
     << "zeroTolerance    : " << zeroTol << "\n"
     << "oneTolerance     : " << oneTol << "\n"
     << "matrixLoadMode   : ";

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

  return os << CYAN("================================\n");
}

void CUDAKernelManager::emitPTX(OptimizationLevel optLevel, int verbose) {
  assert(nWorkerThreads_ > 0);

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  applyLLVMOptimization(
      nWorkerThreads_, optLevel, /* progressBar */ verbose > 0);

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
  std::string archString = "sm_76";
#endif

  const auto createTargetMachine = [&]() -> TargetMachine* {
    return target->createTargetMachine(
        targetTriple, archString, "", {}, std::nullopt);
  };

  // Prepare modules (DL, verify)
  for (auto& pair : llvmContextModulePairs) {
    llvm::Module* mod = pair.llvmModule.get();
    mod->setTargetTriple(targetTriple);
    mod->setDataLayout(createTargetMachine()->createDataLayout());
    if (llvm::verifyModule(*mod, &llvm::errs())) {
      llvm::errs() << "Module verification failed, attempting to proceed with "
                      "PTX emission\n";
    }
  }

  utils::TaskDispatcher dispatcher(nWorkerThreads_);

  for (auto& kernel : standaloneKernels_) {
    dispatcher.enqueue([&]() {
      raw_svector_ostream sstream(kernel->ptxString);
      legacy::PassManager passManager;
      if (createTargetMachine()->addPassesToEmitFile(
              passManager, sstream, nullptr, CodeGenFileType::AssemblyFile)) {
        llvm::errs() << "The target machine can't emit a file of this type\n";
        return;
      }
      passManager.run(*(kernel->llvmModule));
    });
  }
  for (auto& [name, kernels] : graphKernels_) {
    for (auto& kernel : kernels) {
      dispatcher.enqueue([&]() {
        raw_svector_ostream sstream(kernel->ptxString);
        legacy::PassManager passManager;
        if (createTargetMachine()->addPassesToEmitFile(
                passManager, sstream, nullptr, CodeGenFileType::AssemblyFile)) {
          llvm::errs() << "The target machine can't emit a file of this type\n";
          return;
        }
        passManager.run(*(kernel->llvmModule));
      });
    }
  }

  if (verbose > 0)
    std::cerr << "Emitting PTX codes...\n";
  dispatcher.sync(/* progressBar */ verbose > 0);
  jitState = JIT_PTXEmitted;
}

void CUDAKernelManager::initCUJIT(int verbose) {
  assert(jitState == JIT_PTXEmitted);
  assert(nWorkerThreads_ > 0);

  size_t nKernels = getAllStandaloneKernels().size();
  if (nKernels == 0) {
    std::cerr << RED("[Error] ") << "No kernels to JIT.\n";
    return;
  }

  cuInit(0);
  /* cuDeviceGet expects logical index.
   * So if CUDA_VISIBLE_DEVICES="2,3", cuDeviceGet(&cuDevice, 0) selects
   * physical device 2.
   * Therefore, we always choose deviceIdx to be 0, and ask users to control
   * via environment variable CUDA_VISIBLE_DEVICES
   */
  int deviceIdx = 0;
  CUdevice cuDevice;
  CU_CALL(cuDeviceGet(&cuDevice, deviceIdx), "Get CUDA device");

  // Create CUDA contexts
  cuContexts.resize(nWorkerThreads_, nullptr);
  for (unsigned t = 0; t < nWorkerThreads_; ++t) {
    CU_CALL(cuCtxCreate(&cuContexts[t], 0, cuDevice), "Create CUDA context");
  }

  utils::TaskDispatcher dispatcher(nWorkerThreads_);
  std::vector<size_t> sharedMemValues(nKernels);

  for (unsigned i = 0; i < nKernels; ++i) {
    // capture values needed per kernel
    auto& kernel = getAllStandaloneKernels()[i];
    // TODO: Currently ptxString is captured by value. This seems to be due
    // to the property of llvm::SmallVector<char, 0> -- calling str() returns an
    // empty StringRef. One fix is to replace PTXStringType from
    // SmallVector<char, 0> to std::string. Then we need to adjust emitPTX
    // accordingly.
    std::string ptxString(kernel->ptxString.str());
    CUcontext* cuContextPtr = &(kernel->cuTuple.cuContext);
    CUmodule* cuModulePtr = &(kernel->cuTuple.cuModule);
    CUfunction* cuFunctionPtr = &(kernel->cuTuple.cuFunction);
    const char* funcName = kernel->llvmFuncName.c_str();

    dispatcher.enqueue([=, this, &dispatcher]() {
      auto workerID = dispatcher.getWorkerID();

      CU_CHECK(cuCtxSetCurrent(cuContexts[workerID]));
      *cuContextPtr = cuContexts[workerID];
      CU_CHECK(cuModuleLoadData(cuModulePtr, ptxString.c_str()));
      CU_CHECK(cuModuleGetFunction(cuFunctionPtr, *cuModulePtr, funcName));
    });
  }

  if (verbose > 0)
    std::cerr << "Loading CUDA Modules...\n";
  dispatcher.sync(/* progressBar */ verbose > 0);

  jitState = JIT_CUFunctionLoaded;
}

// void CUDAKernelManager::initCUJIT_New(int cuOptLevel, int verbose) {
//   assert(jitState == JIT_PTXEmitted);
//   assert(nWorkerThreads_ > 0);

//   CU_CHECK(cuInit(0));
//   /* cuDeviceGet expects logical index.
//    * So if CUDA_VISIBLE_DEVICES="2,3", cuDeviceGet(&cuDevice, 0) selects
//    * physical device 2.
//    * Therefore, we always choose deviceIdx to be 0, and ask users to control
//    * via environment variable CUDA_VISIBLE_DEVICES
//    */
//   int deviceIdx = 0;
//   CUdevice cuDevice;
//   CU_CHECK(cuDeviceGet(&cuDevice, deviceIdx));
//   CUcontext cuCtx;
//   CU_CHECK(cuDevicePrimaryCtxRetain(&cuCtx, cuDevice));

//   utils::TaskDispatcher dispatcher(nWorkerThreads_);
//   for (auto& kernel : getAllStandaloneKernels()) {
//     std::string ptxString(kernel->ptxString.str());

//     // Capture ptx string by value
//     dispatcher.enqueue([&, ptx = ptxString, cuCtx = cuCtx]() {
//       CU_CHECK(cuCtxSetCurrent(cuCtx));
//       // --- JIT compile (PTX -> CUBIN) ---
//       CUlinkState link;
//       std::vector<CUjit_option> opts;
//       std::vector<void*> vals;
//       // Use device from current context
//       opts.push_back(CU_JIT_TARGET_FROM_CUCONTEXT);
//       vals.push_back(nullptr);

//       // Lower opt level compiles faster (0..4)
//       opts.push_back(CU_JIT_OPTIMIZATION_LEVEL);
//       vals.push_back((void*)(uintptr_t)cuOptLevel);

//       // Optional: collect logs (helpful if something fails)
//       char infoLog[1 << 16] = {0}, errLog[1 << 16] = {0};
//       opts.push_back(CU_JIT_INFO_LOG_BUFFER);
//       vals.push_back(infoLog);
//       opts.push_back(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES);
//       vals.push_back((void*)(uintptr_t)sizeof(infoLog));
//       opts.push_back(CU_JIT_ERROR_LOG_BUFFER);
//       vals.push_back(errLog);
//       opts.push_back(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);
//       vals.push_back((void*)(uintptr_t)sizeof(errLog));

//       CU_CHECK(
//           cuLinkCreate((unsigned)opts.size(), opts.data(), vals.data(),
//           &link));

//       CU_CHECK(cuLinkAddData(link,
//                              CU_JIT_INPUT_PTX,
//                              (void*)ptx.data(),
//                              ptx.size(),
//                              "unit.ptx",
//                              0,
//                              nullptr,
//                              nullptr));

//       void* cubin = nullptr;
//       size_t cubinSize = 0;
//       CUresult c = cuLinkComplete(link, &cubin, &cubinSize);
//       if (c != CUDA_SUCCESS) {
//         fprintf(
//             stderr, "Link failed:\nINFO:\n%s\nERROR:\n%s\n", infoLog,
//             errLog);
//         std::exit(1);
//       }
//     });
//   }
// }

const CUDAKernelInfo*
CUDAKernelManager::getKernelByName(const std::string& llvmFuncName) const {
  for (const auto& kernel : standaloneKernels_) {
    if (kernel->llvmFuncName == llvmFuncName)
      return kernel.get();
  }
  for (const auto& [graphName, kernels] : graphKernels_) {
    for (const auto& kernel : kernels) {
      if (kernel->llvmFuncName == llvmFuncName)
        return kernel.get();
    }
  }
  return nullptr;
}

void CUDAKernelManager::dumpPTX(std::ostream& os,
                                const std::string& llvmFuncName) const {
  const auto* kernelInfo = getKernelByName(llvmFuncName);
  if (!kernelInfo) {
    os << RED("[Error] ") << "Kernel with name '" << llvmFuncName
       << "' not found.\n";
    return;
  }
  os << kernelInfo->ptxString.str().str() << "\n";
}

void CUDAKernelManager::launchCUDAKernel(void* dData,
                                         int nQubits,
                                         const CUDAKernelInfo& kernelInfo,
                                         int blockDim) {
  assert(dData != nullptr);
  assert(kernelInfo.cuTuple.cuContext && kernelInfo.cuTuple.cuFunction);
  cuCtxSetCurrent(kernelInfo.cuTuple.cuContext);

  unsigned nCombos = 1U << (nQubits - kernelInfo.gate->nQubits());
  unsigned gridDim = (nCombos + blockDim - 1) / blockDim;

  // the second arg is supposed to be &combos
  void* kernelParams[2] = {&dData, &nCombos};

  // clang-format off
  CU_CALL(cuLaunchKernel(kernelInfo.cuTuple.cuFunction,
                         gridDim, 1, 1,
                         blockDim, 1, 1,
                         /*sharedMemBytes*/ 0,
                         /*stream*/ 0,
                         kernelParams,
                         nullptr),
          "cuLaunchKernel");
  // clangt-format on
}

void CUDAKernelManager::launchCUDAKernelParam(
    void* dData, // pointer to device statevector
    int nQubits,
    const CUDAKernelInfo& kernelInfo,
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

  void* kernelParams[] = {&dData, &dMatPtr, &combos};

  // clang-format off
  CU_CALL(cuLaunchKernel(kernelInfo.cuTuple.cuFunction,
                         gridDimX, 1, 1,                    // grid dim
                         TILE, 1, 1,                        // block dim
    0,
                         0,                                 // stream
                         kernelParams,                      // kernel arguments
                         nullptr),
          "launchCUDAKernelParam");
  // clang-format on
}

#undef DEBUG_TYPE