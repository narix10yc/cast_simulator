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
     << "forceDenseKernel : " << forceDenseKernel << "\n"
     << "zeroTolerance    : " << zeroTol << "\n"
     << "oneTolerance     : " << oneTol << "\n"
     << "assumeContiguousTargets : " << assumeContiguousTargets << "\n"
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

  // Build the index matching modules -> kernels
  rebuildOrderedKernelIndex_();
  assert(orderedKernels_.size() == llvmContextModulePairs.size() &&
         "Mismatch between modules and kernels");

  utils::TaskDispatcher dispatcher(nWorkerThreads_);

  for (unsigned i = 0; i < orderedKernels_.size(); i++) {
    dispatcher.enqueue([&, i]() {
      raw_svector_ostream sstream(orderedKernels_[i]->ptxString);
      legacy::PassManager passManager;
      if (createTargetMachine()->addPassesToEmitFile(
              passManager, sstream, nullptr, CodeGenFileType::AssemblyFile)) {
        llvm::errs() << "The target machine can't emit a file of this type\n";
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

void CUDAKernelManager::initCUJIT(int verbose) {
  assert(jitState == JIT_PTXEmitted);
  assert(nWorkerThreads_ > 0);

  if (orderedKernels_.empty())
    rebuildOrderedKernelIndex_();

  size_t nKernels = orderedKernels_.size();
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
    auto* kernel = orderedKernels_[i];
    // TODO: Currently ptxString is captured by value. This seems to be due to
    // the property of llvm::SmallVector<char, 0> -- calling str() returns an
    // empty StringRef. One fix is to replace PTXStringType from
    // SmallVector<char, 0> to std::string. Then we need to adjust emitPTX
    // accordingly.
    std::string ptxString(kernel->ptxString.str());
    CUcontext* cuContextPtr = &(kernel->cuTuple.cuContext);
    CUmodule* cuModulePtr = &(kernel->cuTuple.cuModule);
    CUfunction* cuFunctionPtr = &(kernel->cuTuple.cuFunction);
    const char* funcName = kernel->llvmFuncName.c_str();

    dispatcher.enqueue([=, this, &sharedMemValues, &dispatcher]() {
      auto workerID = dispatcher.getWorkerID();

      CU_CALL(cuCtxSetCurrent(cuContexts[workerID]), "cuCtxSetCurrent");
      *cuContextPtr = cuContexts[workerID];
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
    std::cerr << "Loading CUDA Modules...\n";
  dispatcher.sync(/* progressBar */ verbose > 0);

  for (unsigned i = 0; i < nKernels; ++i) {
    orderedKernels_[i]->cuTuple.sharedMemBytes = sharedMemValues[i];
  }

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

void CUDAKernelManager::launchCUDAKernel(void* dData,
                                         int nQubits,
                                         const CUDAKernelInfo& kernelInfo,
                                         int /*ignored*/) {
  assert(dData != nullptr);
  assert(kernelInfo.cuTuple.cuContext && kernelInfo.cuTuple.cuFunction);
  cuCtxSetCurrent(kernelInfo.cuTuple.cuContext);

  unsigned k = kernelInfo.gate->nQubits();
  unsigned N = 1u << k;
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
  unsigned tilesPerGate =
      (N + TILE - 1) / TILE; // used by shared‑tiled & row‑per‑lane

  // device limits / targets
  CUdevice dev;
  cuCtxGetDevice(&dev);
  int sms = 0;
  cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);

  int maxBlocksPerSMAttr = 0;
  cuDeviceGetAttribute(&maxBlocksPerSMAttr,
                       CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
                       dev);
  unsigned maxCTAsPerSMHW =
      (maxBlocksPerSMAttr > 0) ? (unsigned)maxBlocksPerSMAttr : 32u;

  // derive warps/CTA
  unsigned warpsPerCTA = (kernelInfo.warpsPerCTA > 0) ? kernelInfo.warpsPerCTA
                                                      : std::max(1u, BLK / 32u);

  // targets (env‑overridable)
  // default 48; override with CAST_WARPS_PER_SM
  unsigned targetWarpsPerSM = 48u;
  if (const char* envW = std::getenv("CAST_WARPS_PER_SM")) {
    unsigned v = (unsigned)std::strtoul(envW, nullptr, 10);
    if (v)
      targetWarpsPerSM = v;
  }
  unsigned targetCTAsPerSM = std::min(
      std::max(1u, (targetWarpsPerSM + warpsPerCTA - 1u) / warpsPerCTA), // ceil
      maxCTAsPerSMHW);
  if (const char* env = std::getenv("CAST_CTA_PER_SM")) {
    unsigned v = (unsigned)std::strtoul(env, nullptr, 10);
    if (v) {
      targetCTAsPerSM = std::min(std::max(1u, v), maxCTAsPerSMHW);
    }
  }

  const unsigned maxComboSlots = std::max(1u, (unsigned)sms * targetCTAsPerSM);

  // --- pick grid.x depending on style ---------------------------------------
  unsigned gridX = 1;

  if (kernelInfo.kstyle == "imm-inline-warp") {
    // Lane‑per‑combo inline: grid * warps * 32 ≳ combos, no tilesPerGate factor
    unsigned blocksForWork =
        (combos + warpsPerCTA * 32u - 1u) / (warpsPerCTA * 32u);
    gridX = std::max(1u, std::min(blocksForWork, maxComboSlots));
  } else if (kernelInfo.oneThreadPerBlock) {
    // 1‑thread inline: one combo per CTA
    gridX = std::max(1u, std::min(combos, maxComboSlots));
  } else {
    // Shared‑tiled & row‑per‑lane paths: grid.x must be a multiple of
    // tilesPerGate
    const unsigned gridComboSlots = std::min(combos, maxComboSlots);
    gridX = tilesPerGate * gridComboSlots;
  }

  void* kernelParams[2] = {&dData, &combos};

  // clang-format off
  CU_CALL(cuLaunchKernel(kernelInfo.cuTuple.cuFunction,
                         gridX, 1, 1,
                         BLK, 1, 1,
                         kernelInfo.cuTuple.sharedMemBytes,
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
                         kernelInfo.cuTuple.sharedMemBytes, // shared memory
                         0,                                 // stream
                         kernelParams,                      // kernel arguments
                         nullptr),
          "launchCUDAKernelParam");
  // clang-format on
}

#undef DEBUG_TYPE