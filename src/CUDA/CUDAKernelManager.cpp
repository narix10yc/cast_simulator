#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/Config.h"

#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Config/llvm-config.h"

#include "cast/CUDA/Config.h"
#include "utils/Formats.h"
#include "utils/TaskDispatcher.h"
#include "utils/iocolor.h"
#include "utils/utils.h"
#include <fstream>
#include <cstdint>

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
    {
      using namespace llvm;

      auto rewriteCpAsyncCgToCa = [](Module& M) {
        auto rewriteOne = [&](StringRef bad, StringRef good) {
          if (Function* Bad = M.getFunction(bad)) {
            FunctionCallee GoodC = M.getOrInsertFunction(good, Bad->getFunctionType());
            Function* Good = llvm::cast<Function>(GoodC.getCallee());

            SmallVector<Instruction*, 8> toErase;
            for (User* U : Bad->users()) {
              if (auto* CI = dyn_cast<CallInst>(U)) {
                IRBuilder<> B(CI);
                SmallVector<Value*, 8> Args(CI->arg_begin(), CI->arg_end());
                CallInst* New = B.CreateCall(Good, Args);
                New->setTailCallKind(CI->getTailCallKind());
                New->setCallingConv(CI->getCallingConv());
                New->setAttributes(CI->getAttributes());
                New->setDebugLoc(CI->getDebugLoc());
                toErase.push_back(CI);
              } else if (auto* II = dyn_cast<InvokeInst>(U)) {
                SmallVector<Value*, 8> Args(II->arg_begin(), II->arg_end());
                InvokeInst* NewI = InvokeInst::Create(
                    Good, II->getNormalDest(), II->getUnwindDest(), Args, "", II);
                NewI->setCallingConv(II->getCallingConv());
                NewI->setAttributes(II->getAttributes());
                NewI->setDebugLoc(II->getDebugLoc());
                toErase.push_back(II);
              }
            }
            for (Instruction* I : toErase) I->eraseFromParent();
            if (Bad->use_empty()) Bad->eraseFromParent();
          }
        };
        rewriteOne("llvm.nvvm.cp.async.cg.shared.global.4",
                  "llvm.nvvm.cp.async.ca.shared.global.4");
        rewriteOne("llvm.nvvm.cp.async.cg.shared.global.8",
                  "llvm.nvvm.cp.async.ca.shared.global.8");
      };

      rewriteCpAsyncCgToCa(*mod);
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
      CU_CALL(cuCtxSetCurrent(cuContexts[0]), "cuCtxSetCurrent");
      *cuContextPtr = cuContexts[0];

      auto loadModuleWithLogs = [&](CUmodule* module, const char* ptx) -> CUresult {
        char infoLog[8192]  = {0};
        char errorLog[8192] = {0};
        // sizes must be values, not pointers
        unsigned int infoSize  = sizeof(infoLog);
        unsigned int errorSize = sizeof(errorLog);

        CUjit_option opts[] = {
          CU_JIT_TARGET_FROM_CUCONTEXT,
          CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
          CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
          CU_JIT_LOG_VERBOSE
        };

        void* optvals[] = {
          nullptr,
          (void*)infoLog,                          // buffer pointer
          (void*)(uintptr_t)infoSize,              // SIZE AS VALUE
          (void*)errorLog,                         // buffer pointer
          (void*)(uintptr_t)errorSize,             // SIZE AS VALUE
          (void*)(uintptr_t)1                      // verbose = 1
        };

        unsigned int numOpts = sizeof(opts) / sizeof(opts[0]);

        CUresult res = cuModuleLoadDataEx(module, ptx, numOpts, opts, optvals);

        // make sure buffers are terminated (driver usually does, but be defensive)
        infoLog[sizeof(infoLog) - 1]   = '\0';
        errorLog[sizeof(errorLog) - 1] = '\0';

        if (res != CUDA_SUCCESS) {
          std::cerr << "[CUDA JIT Error] cuModuleLoadDataEx failed with code "
                    << static_cast<int>(res) << "\n";
          if (errorLog[0]) std::cerr << "--- Error log ---\n" << errorLog << "\n";
          if (infoLog[0])  std::cerr << "--- Info log ---\n" << infoLog  << "\n";
        } else if (verbose > 1 && (infoLog[0] || errorLog[0])) {
          std::cerr << "--- Info log ---\n" << infoLog << "\n";
        }
        return res;
      };
      if (loadModuleWithLogs(cuModulePtr, ptxString.c_str()) != CUDA_SUCCESS) {
        // Leave cuFunctionPtr null; later code must not try to launch this kernel.
        return;
      }
      CU_CALL(cuModuleGetFunction(cuFunctionPtr, *cuModulePtr, funcName),
              "cuModuleGetFunction");

      // int staticShared = 0;
      // CU_CALL(cuFuncGetAttribute(&staticShared,
      //                           CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
      //                           *cuFunctionPtr),
      //         "cuFuncGetAttribute(SHARED_SIZE_BYTES)");
      // sharedMemValues[i] = static_cast<size_t>(staticShared);

      // // Pick carveout per style
      // const std::string &style =
      //     kernel->kstyle.empty() ? std::string("") : kernel->kstyle; // field already used in launch
      // int carveout = (style == "imm-shared") ? CU_SHAREDMEM_CARVEOUT_MAX_SHARED
      //                                       : CU_SHAREDMEM_CARVEOUT_MAX_L1;
      // cuFuncSetAttribute(*cuFunctionPtr,
      //                   CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
      //                   carveout);
      int carveout = CU_SHAREDMEM_CARVEOUT_MAX_L1;
      const std::string style = kernel->kstyle;  // requires Fix #1
      if (style == "imm-shared" || style == "imm-shared-warp" || style == "ptr-shared") {
        carveout = CU_SHAREDMEM_CARVEOUT_MAX_SHARED;
      }
      cuFuncSetAttribute(*cuFunctionPtr,
          CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
          carveout);
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
  // assert(kernelInfo.cuTuple.cuContext && kernelInfo.cuTuple.cuFunction);
  if (!kernelInfo.cuTuple.cuContext || !kernelInfo.cuTuple.cuFunction) {
    std::cerr << RED("[Error] ")
              << "Kernel launch skipped: JIT/module/function not available for \""
              << kernelInfo.llvmFuncName << "\"\n";
    return;
  }
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