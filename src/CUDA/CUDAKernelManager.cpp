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
  os << CYAN("=== GPU Kernel Gen Config ===\n") << "precision: f"
     << static_cast<int>(precision) << "\n"
     << "forceDenseKernel : " << forceDenseKernel << "\n"
     << "zeroTolerance : " << zeroTol << "\n"
     << "oneTolerance : " << oneTol << "\n"
     << "assumeContiguousTargets : " << assumeContiguousTargets << "\n"
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

  // Build the index matching modules -> kernels
  rebuildOrderedKernelIndex_();
  assert(orderedKernels_.size() == llvmContextModulePairs.size() &&
         "Mismatch between modules and kernels");

  utils::TaskDispatcher dispatcher(nThreads);

  for (unsigned i = 0; i < orderedKernels_.size(); i++) {
    dispatcher.enqueue([&, i](){
      raw_svector_ostream sstream(orderedKernels_[i]->ptxString);
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

  if (orderedKernels_.empty())
    rebuildOrderedKernelIndex_();

  size_t nKernels = orderedKernels_.size();
  if (nKernels == 0) {
    std::cerr << RED("[Error] ") << "No kernels to JIT.\n";
    return;
  }
  if (nKernels < static_cast<size_t>(nThreads)) {
    std::cerr << YELLOW("[Warning] ") << "Calling initCUJIT with " << nThreads
              << " threads when there are only " << nKernels
              << " kernels. Set nThreads to " << nKernels << " instead.\n";
    nThreads = static_cast<int>(nKernels);
  }

  cuInit(0);
  CUdevice cuDevice;
  int deviceIdx = 0; // honor CUDA_VISIBLE_DEVICES
  CU_CALL(cuDeviceGet(&cuDevice, deviceIdx), "Get CUDA device");

  // Create CUDA contexts
  // cuContexts.resize(nThreads, nullptr);
  // for (unsigned t = 0; t < static_cast<unsigned>(nThreads); ++t) {
  //   CU_CALL(cuCtxCreate(&cuContexts[t], 0, cuDevice), "Create CUDA context");
  // }
  CUcontext sharedCtx = nullptr;
  // Primary context is the simplest way to share:
  CU_CALL(cuDevicePrimaryCtxRetain(&sharedCtx, cuDevice), "Retain primary context");
  cuContexts.clear();
  cuContexts.push_back(sharedCtx);

  utils::TaskDispatcher dispatcher(nThreads);
  std::vector<size_t> sharedMemValues(nKernels);

  for (unsigned i = 0; i < nKernels; ++i) {
    // capture values needed per kernel
    auto* kernel = orderedKernels_[i];
    std::string ptxString(kernel->ptxString.str());
    CUcontext*  cuContextPtr  = &(kernel->cuTuple.cuContext);
    CUmodule*   cuModulePtr   = &(kernel->cuTuple.cuModule);
    CUfunction* cuFunctionPtr = &(kernel->cuTuple.cuFunction);
    const char* funcName      = kernel->llvmFuncName.c_str();

    dispatcher.enqueue([=, &sharedMemValues, this, &dispatcher]() {
      // auto workerID = dispatcher.getWorkerID();
      // CU_CALL(cuCtxSetCurrent(cuContexts[workerID]), "cuCtxSetCurrent");
      // *cuContextPtr = cuContexts[workerID];
      CU_CALL(cuCtxSetCurrent(cuContexts[0]), "cuCtxSetCurrent");
      *cuContextPtr = cuContexts[0];
      CU_CALL(cuModuleLoadData(cuModulePtr, ptxString.c_str()),
              "cuModuleLoadData");
      CU_CALL(cuModuleGetFunction(cuFunctionPtr, *cuModulePtr, funcName),
              "cuModuleGetFunction");
      cuFuncSetAttribute(*cuFunctionPtr,
                   CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                   CU_SHAREDMEM_CARVEOUT_MAX_L1);
      int staticShared = 0;
      CU_CALL(cuFuncGetAttribute(&staticShared,
                                 CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                 *cuFunctionPtr),
              "cuFuncGetAttribute(SHARED_SIZE_BYTES)");
      sharedMemValues[i] = static_cast<size_t>(staticShared);
    });
  }

  if (verbose > 0)
    std::cerr << "Loading PTX codes and getting CUDA functions...\n";
  dispatcher.sync(/* progressBar */ verbose > 0);

  for (unsigned i = 0; i < nKernels; ++i) {
    orderedKernels_[i]->cuTuple.sharedMemBytes = sharedMemValues[i];
  }

  // Optional: dump PTX for debugging
  // for (const auto* kernel : orderedKernels_) {
  //   const std::string fileName = kernel->llvmFuncName + ".ptx";
  //   std::ofstream ofs(fileName, std::ios::out | std::ios::trunc);
  //   ofs << kernel->ptxString.str().str();
  //   if (verbose > 0)
  //     std::cerr << "Wrote " << fileName << '\n';
  // }

  jitState = JIT_CUFunctionLoaded;
}


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

// void CUDAKernelManager::launchCUDAKernel(
//     void* dData, int nQubits, const CUDAKernelInfo& kernelInfo, int /*ignored*/)
// {
//   assert(dData != nullptr);
//   assert(kernelInfo.cuTuple.cuContext && kernelInfo.cuTuple.cuFunction);
//   cuCtxSetCurrent(kernelInfo.cuTuple.cuContext);

//   unsigned k    = kernelInfo.gate->nQubits();
//   unsigned N    = 1u << k;
//   unsigned TILE = std::min(256u, N);

//   constexpr unsigned MIN_BLOCK_THREADS = 32;

//   unsigned BLK;
//   if (kernelInfo.oneThreadPerBlock) {
//     BLK = 1u;
//   } else if (kernelInfo.warpsPerCTA > 0) {
//     BLK = kernelInfo.warpsPerCTA * 32u;      // WPR path ⇒ multiple of 32
//   } else {
//     // legacy shared‑tiled path (still supported)
//     BLK = std::max(TILE, MIN_BLOCK_THREADS); // >= TILE for old kernel
//   }

//   // unsigned combos       = 1u << (nQubits - k);
//   // unsigned tilesPerGate = (N + TILE - 1) / TILE;

//   // dim3 blockDim(BLK, 1, 1);
//   // dim3 gridDim(combos * tilesPerGate, 1, 1);

//   unsigned combos       = 1u << (nQubits - k);
//   unsigned tilesPerGate = (N + TILE - 1) / TILE;

//   // persistent grid sizing
//   // CUdevice dev; cuCtxGetDevice(&dev);
//   // int sms=0; cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
//   // const unsigned targetCTAsPerSM = 8;                         // good default on GA102
//   // const unsigned maxComboSlots   = std::max(1u, sms * targetCTAsPerSM);
//   // const unsigned gridComboSlots  = std::min(combos, maxComboSlots);
//   // const unsigned gridX           = tilesPerGate * gridComboSlots; // multiple of tilesPerGate

//   // dim3 blockDim(BLK, 1, 1);
//   // dim3 gridDim(gridX, 1, 1);

//   // persistent grid sizing (adaptive)
//   CUdevice dev; cuCtxGetDevice(&dev);

//   int sms = 0;
//   cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);

//   // Query HW max blocks per SM to avoid hard-coding (32 on Ampere desktop)
//   int maxBlocksPerSMAttr = 0;
//   cuDeviceGetAttribute(&maxBlocksPerSMAttr,
//                       CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, dev);
//   unsigned maxCTAsPerSMHW = (maxBlocksPerSMAttr > 0)
//                             ? static_cast<unsigned>(maxBlocksPerSMAttr) : 32u;

//   // Derive warps per CTA from kernel style / chosen block size (BLK)
//   unsigned warpsPerCTA = 0;
//   if (kernelInfo.oneThreadPerBlock) {
//     warpsPerCTA = 1;                    // 1 thread still occupies 1 warp slot
//   } else if (kernelInfo.warpsPerCTA > 0) {
//     warpsPerCTA = kernelInfo.warpsPerCTA;   // WPR path provides this explicitly
//   } else {
//     warpsPerCTA = std::max(1u, BLK / 32u);  // legacy tiled: BLK is multiple of 32
//   }

//   // Target active warps/SM. 32–64 is a good range on GA102; default 48.
//   // Optional override via env: CAST_WARPS_PER_SM=<N>
//   unsigned targetWarpsPerSM = 48u;
//   if (const char* envW = std::getenv("CAST_WARPS_PER_SM")) {
//     unsigned v = static_cast<unsigned>(std::strtoul(envW, nullptr, 10));
//     if (v) targetWarpsPerSM = v;
//   }

//   // Compute target CTAs/SM, clamp to HW limit.
//   // Optional override via env: CAST_CTA_PER_SM=<N>
//   unsigned targetCTAsPerSM =
//       (targetWarpsPerSM + warpsPerCTA - 1u) / warpsPerCTA;   // ceil division
//   targetCTAsPerSM = std::min(targetCTAsPerSM, maxCTAsPerSMHW);

//   if (const char* env = std::getenv("CAST_CTA_PER_SM")) {
//     unsigned v = static_cast<unsigned>(std::strtoul(env, nullptr, 10));
//     if (v) {
//       if (v < 1u) v = 1u;
//       if (v > maxCTAsPerSMHW) v = maxCTAsPerSMHW;
//       targetCTAsPerSM = v;
//     }
//   }

//   const unsigned maxComboSlots  = std::max(1u, static_cast<unsigned>(sms) * targetCTAsPerSM);
//   const unsigned gridComboSlots = std::min(combos, maxComboSlots);
//   const unsigned gridX          = tilesPerGate * gridComboSlots;

//   dim3 blockDim(BLK, 1, 1);
//   dim3 gridDim(gridX, 1, 1);

//   const void* cMatPtr = nullptr;
//   if (auto* stdQuGate = llvm::dyn_cast_or_null<StandardQuantumGate>(kernelInfo.gate.get())) {
//     if (auto scalarGM = stdQuGate->getScalarGM()) cMatPtr = scalarGM->matrix().data();
//   }
//   // void* kernelParams[2] = { &dData, &cMatPtr };
//   void* kernelParams[2] = { &dData, &combos };

//   // (optional debug)
//   // fprintf(stderr,"[Launch] k=%u style=%s TILE=%u block=(%u,1,1) grid=(%u,1,1)\n",
//   //         k, kernelInfo.oneThreadPerBlock ? "inline" :
//   //             (kernelInfo.warpsPerCTA?"wpr":"legacy"),
//   //         TILE, BLK, gridDim.x);

//   CU_CALL(cuLaunchKernel(kernelInfo.cuTuple.cuFunction,
//                          gridDim.x, 1, 1,
//                          blockDim.x, 1, 1,
//                          kernelInfo.cuTuple.sharedMemBytes,
//                          /*stream*/ 0, kernelParams, nullptr),
//           "cuLaunchKernel");
// }

void CUDAKernelManager::launchCUDAKernel(
    void* dData, int nQubits, const CUDAKernelInfo& kernelInfo, int /*ignored*/)
{
  assert(dData != nullptr);
  assert(kernelInfo.cuTuple.cuContext && kernelInfo.cuTuple.cuFunction);
  cuCtxSetCurrent(kernelInfo.cuTuple.cuContext);

  unsigned k    = kernelInfo.gate->nQubits();
  unsigned N    = 1u << k;
  unsigned TILE = std::min(256u, N);

  constexpr unsigned MIN_BLOCK_THREADS = 32;
  unsigned BLK;
  if (kernelInfo.oneThreadPerBlock) {
    BLK = 1u;
  } else if (kernelInfo.warpsPerCTA > 0) {
    BLK = kernelInfo.warpsPerCTA * 32u; // multiple of warp
  } else {
    BLK = std::max(TILE, MIN_BLOCK_THREADS); // shared‑tiled
  }

  // work geometry
  unsigned combos = 1u << (nQubits - k);
  unsigned tilesPerGate = (N + TILE - 1) / TILE;  // used by shared‑tiled & row‑per‑lane

  // device limits / targets
  CUdevice dev; cuCtxGetDevice(&dev);
  int sms = 0; cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);

  int maxBlocksPerSMAttr = 0;
  cuDeviceGetAttribute(&maxBlocksPerSMAttr,
                       CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, dev);
  unsigned maxCTAsPerSMHW = (maxBlocksPerSMAttr > 0)
                            ? (unsigned)maxBlocksPerSMAttr : 32u;

  // derive warps/CTA
  unsigned warpsPerCTA = (kernelInfo.warpsPerCTA > 0) ? kernelInfo.warpsPerCTA
                                                      : std::max(1u, BLK / 32u);

  // targets (env‑overridable)
  unsigned targetWarpsPerSM = 48u;  // default 48; override with CAST_WARPS_PER_SM
  if (const char* envW = std::getenv("CAST_WARPS_PER_SM")) {
    unsigned v = (unsigned)std::strtoul(envW, nullptr, 10);
    if (v) targetWarpsPerSM = v;
  }
  unsigned targetCTAsPerSM = std::min(
      std::max(1u, (targetWarpsPerSM + warpsPerCTA - 1u) / warpsPerCTA), // ceil
      maxCTAsPerSMHW);
  if (const char* env = std::getenv("CAST_CTA_PER_SM")) {
    unsigned v = (unsigned)std::strtoul(env, nullptr, 10);
    if (v) { targetCTAsPerSM = std::min(std::max(1u, v), maxCTAsPerSMHW); }
  }

  const unsigned maxComboSlots = std::max(1u, (unsigned)sms * targetCTAsPerSM);

  // --- pick grid.x depending on style ---------------------------------------
  unsigned gridX = 1;

  if (kernelInfo.kstyle == "imm-inline-warp") {
    // Lane‑per‑combo inline: grid * warps * 32 ≳ combos, no tilesPerGate factor
    unsigned blocksForWork = (combos + warpsPerCTA*32u - 1u) / (warpsPerCTA*32u);
    gridX = std::max(1u, std::min(blocksForWork, maxComboSlots));
  }
  else if (kernelInfo.oneThreadPerBlock) {
    // 1‑thread inline: one combo per CTA
    gridX = std::max(1u, std::min(combos, maxComboSlots));
  }
  else {
    // Shared‑tiled & row‑per‑lane paths: grid.x must be a multiple of tilesPerGate
    const unsigned gridComboSlots = std::min(combos, maxComboSlots);
    gridX = tilesPerGate * gridComboSlots;
  }

  dim3 blockDim(BLK, 1, 1);
  dim3 gridDim(gridX, 1, 1);

  void* kernelParams[2] = { &dData, &combos };

  CU_CALL(cuLaunchKernel(kernelInfo.cuTuple.cuFunction,
                         gridDim.x, 1, 1,
                         blockDim.x, 1, 1,
                         kernelInfo.cuTuple.sharedMemBytes,
                         /*stream*/ 0, kernelParams, nullptr),
          "cuLaunchKernel");
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

  dim3 gridDim(gridDimX, 1, 1);
  dim3 blockDim(TILE, 1, 1);

  void* kernelParams[] = { &dData, &dMatPtr, &combos };

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