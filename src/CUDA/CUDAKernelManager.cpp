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

std::ostream& CUDAKernelInfo::displayInfo(std::ostream& os) const {
  os << CYAN("=== Info of CUDA Kernel @ " << (void*)this << " ===\n")
     << "Function Name    : " << llvmFuncName << "\n"
     << "Precision        : f" << static_cast<int>(precision) << "\n"
     << "Gate             : " << gate.get() << "\n"
     << "PTX              : ";
  if (ptxString.empty())
    os << "No\n";
  else
    os << "Yes, with size " << utils::fmt_mem(ptxString.size()) << "\n";

  os << "CUBIN            : ";
  if (cubinData.empty())
    os << "No\n";
  else
    os << "Yes, with size " << utils::fmt_mem(cubinData.size()) << "\n";

  os << "LLVM Context     : " << llvmContext << "\n"
     << "LLVM Module      : " << llvmModule << "\n"
     << "CU Context       : " << cuContext << "\n"
     << "CU Module        : " << cuModule << "\n"
     << "CU Function      : " << cuFunction << "\n";
  return os << CYAN("================================\n");
}

std::ostream& CUDAKernelManager::displayInfo(std::ostream& os) const {
  os << CYAN("=== Info of CUDA Kernel Manager @ " << (void*)this << " ===\n")
     << "Num Worker Threads : " << nWorkerThreads_ << "\n"
     << "JIT State          : " << static_cast<int>(jitState) << "\n"
     << "Primary CU Context : " << primaryCuCtx << "\n"
     << "Num CU Modules     : " << cuModules.size() << "\n";

  int nKernels = 0;
  size_t totalPTXSize = 0, totalCUBINSize = 0;
  for (const auto& kernel : *this) {
    ++nKernels;
    totalPTXSize += kernel.ptxString.size();
    totalCUBINSize += kernel.cubinData.size();
  }
  os << "Num Kernels        : " << nKernels << "\n"
     << "Total PTX Size     : " << utils::fmt_mem(totalPTXSize) << "\n"
     << "Total CUBIN Size   : " << utils::fmt_mem(totalCUBINSize) << "\n";
  return os << CYAN("================================\n");
}

namespace {

class raw_pwrite_vector_ostream : public llvm::raw_pwrite_stream {
  std::string& out;

  void write_impl(const char* Ptr, size_t Size) override {
    out.append(Ptr, Size);
  }

  void pwrite_impl(const char* Ptr, size_t Size, uint64_t Offset) override {
    if (out.size() < Offset + Size)
      out.resize(static_cast<size_t>(Offset + Size));
    std::memcpy(out.data() + Offset, Ptr, Size);
  }

  uint64_t current_pos() const override { return out.size(); }

public:
  explicit raw_pwrite_vector_ostream(std::string& str) : out(str) {
    SetUnbuffered();
  }
  ~raw_pwrite_vector_ostream() override { flush(); }
};
} // namespace

void CUDAKernelManager::compileLLVMIRToPTX(int llvmOptLevel, int verbose) {
  assert(nWorkerThreads_ > 0);

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  llvm::OptimizationLevel optLevel;
  switch (llvmOptLevel) {
  case 0:
    optLevel = llvm::OptimizationLevel::O0;
    break;
  case 1:
    optLevel = llvm::OptimizationLevel::O1;
    break;
  case 2:
    optLevel = llvm::OptimizationLevel::O2;
    break;
  case 3:
    optLevel = llvm::OptimizationLevel::O3;
    break;
  default:
    assert(false && "Invalid LLVM optimization level");
    // Default to O1
    optLevel = llvm::OptimizationLevel::O1;
    break;
  }

  applyLLVMOptimization(nWorkerThreads_, optLevel, verbose > 0);

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

  for (auto& kernel : *this) {
    dispatcher.enqueue([&]() {
      raw_pwrite_vector_ostream vecStream(kernel.ptxString);
      legacy::PassManager passManager;
      if (createTargetMachine()->addPassesToEmitFile(
              passManager, vecStream, nullptr, CodeGenFileType::AssemblyFile)) {
        llvm::errs() << "The target machine can't emit a file of this type\n";
        return;
      }
      passManager.run(*(kernel.llvmModule));
    });
  }

  if (verbose > 0)
    std::cerr << "Generating PTX codes...\n";
  dispatcher.sync(/* progressBar */ verbose > 0);
  jitState = JIT_PTXEmitted;
}

static void compileToCubin_work(int cuOptLevel, CUDAKernelInfo& kernel) {
  assert(kernel.cuContext != nullptr);
  assert(!kernel.ptxString.empty());

  CU_CHECK(cuCtxSetCurrent(kernel.cuContext));
  // JIT/link options
  constexpr size_t LOG_SZ = 1 << 15;
  std::vector<char> info(LOG_SZ, 0), err(LOG_SZ, 0);
  float wallTime = 0.0f;

  unsigned int fastCompile = 1;
  unsigned int targetFromCtx = 1;

  CUjit_option options[] = {CU_JIT_OPTIMIZATION_LEVEL,
                            CU_JIT_FAST_COMPILE,
                            CU_JIT_TARGET_FROM_CUCONTEXT,
                            CU_JIT_INFO_LOG_BUFFER,
                            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                            CU_JIT_ERROR_LOG_BUFFER,
                            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                            CU_JIT_WALL_TIME,
                            CU_JIT_LOG_VERBOSE};

  void* optionVals[] = {(void*)(uintptr_t)cuOptLevel,
                        (void*)(uintptr_t)fastCompile,
                        (void*)(uintptr_t)targetFromCtx,
                        info.data(),
                        (void*)(uintptr_t)LOG_SZ,
                        err.data(),
                        (void*)(uintptr_t)LOG_SZ,
                        &wallTime,
                        (void*)(uintptr_t)1};

  CUlinkState linkState = nullptr;

  CU_CHECK(cuLinkCreate((unsigned int)(sizeof(options) / sizeof(options[0])),
                        options,
                        optionVals,
                        &linkState));

  // PTX MUST be NUL-terminated
  CU_CHECK(cuLinkAddData(linkState,
                         CU_JIT_INPUT_PTX,
                         (void*)kernel.ptxString.c_str(),
                         kernel.ptxString.size() + 1,
                         kernel.llvmFuncName.c_str(),
                         0,
                         nullptr,
                         nullptr));

  void* cubinOut = nullptr;
  size_t cubinSize = 0;
  CU_CHECK(cuLinkComplete(linkState, &cubinOut, &cubinSize));

  // copy cubinOut to the kernel info
  // cubinOut is owned by the link state, so will be invalidated after
  // calling cuLinkDestroy
  kernel.cubinData.assign(static_cast<uint8_t*>(cubinOut),
                          static_cast<uint8_t*>(cubinOut) + cubinSize);

  cuLinkDestroy(linkState);
}

static void loadCubin_work(CUDAKernelInfo& kernel) {
  assert(kernel.cuContext != nullptr);
  assert(!kernel.cubinData.empty());

  CU_CHECK(cuCtxSetCurrent(kernel.cuContext));

  CU_CHECK(cuModuleLoadData(&kernel.cuModule, kernel.cubinData.data()));
  CU_CHECK(cuModuleGetFunction(
      &kernel.cuFunction, kernel.cuModule, kernel.llvmFuncName.c_str()));
}

void CUDAKernelManager::compilePTXToCubin(int cuOptLevel, int verbose) {
  assert(jitState == JIT_PTXEmitted);
  assert(nWorkerThreads_ > 0);
  assert(0 <= cuOptLevel && cuOptLevel <= 4);

  CU_CHECK(cuInit(0));
  /* cuDeviceGet expects logical index.
   * So if CUDA_VISIBLE_DEVICES="2,3", cuDeviceGet(&cuDevice, 0) selects
   * physical device 2.
   * Therefore, we always choose deviceIdx to be 0, and ask users to control
   * via environment variable CUDA_VISIBLE_DEVICES
   */
  int deviceIdx = 0;
  CUdevice cuDevice;
  CU_CHECK(cuDeviceGet(&cuDevice, deviceIdx));
  // create primary cuda context
  CU_CHECK(cuCtxCreate(&primaryCuCtx, 0, cuDevice));
  CU_CHECK(cuDevicePrimaryCtxRetain(&primaryCuCtx, cuDevice));

  utils::TaskDispatcher dispatcher(nWorkerThreads_);
  for (auto& kernel : *this) {
    dispatcher.enqueue([=, this, &kernel]() {
      // must set cuContext before calling compileToCubin_work
      kernel.cuContext = primaryCuCtx;
      compileToCubin_work(cuOptLevel, kernel);
    });
  }

  if (verbose > 0)
    std::cerr << "JIT Compile PTX to CUBIN...\n";
  dispatcher.sync(/* progressBar */ verbose > 0);

  jitState = JIT_CUBIN;
}

void CUDAKernelManager::loadCubin(int verbose) {
  assert(jitState == JIT_CUBIN);
  assert(nWorkerThreads_ > 0);

  utils::TaskDispatcher dispatcher(nWorkerThreads_);

  // this->cuModules takes the ownership of all cuda modules created in
  // loadCubin_work
  cuModules.resize(this->numKernels());
  CUmodule* cuModulePtr = cuModules.data();
  for (auto& kernel : *this) {
    dispatcher.enqueue([=, this, &kernel]() {
      loadCubin_work(kernel);
      *cuModulePtr = kernel.cuModule;
    });
    ++cuModulePtr;
  }

  if (verbose > 0)
    std::cerr << "Load CUBIN...\n";
  dispatcher.sync(/* progressBar */ verbose > 0);
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
  os << kernelInfo->ptxString << "\n";
}

void CUDAKernelManager::launchCUDAKernel(void* dData,
                                         int nQubits,
                                         const CUDAKernelInfo& kernelInfo,
                                         int blockDim) {
  assert(dData != nullptr);
  assert(kernelInfo.cuContext && kernelInfo.cuFunction);

  unsigned nCombos = 1U << (nQubits - kernelInfo.gate->nQubits());
  unsigned gridDim = (nCombos + blockDim - 1) / blockDim;

  // the second arg is supposed to be &combos
  void* kernelParams[2] = {&dData, &nCombos};

  cuCtxSetCurrent(kernelInfo.cuContext);
  // clang-format off
  CU_CALL(cuLaunchKernel(kernelInfo.cuFunction,
                         gridDim, 1, 1,
                         blockDim, 1, 1,
                         /*sharedMemBytes*/ 0,
                         /*stream*/ 0,
                         kernelParams,
                         nullptr),
          "cuLaunchKernel");
  // clangt-format on
}

// void CUDAKernelManager::launchCUDAKernelParam(
//     void* dData, // pointer to device statevector
//     int nQubits,
//     const CUDAKernelInfo& kernelInfo,
//     void* dMatPtr, // pointer to device matrix
//     int blockSize  // ignored if fixed TILE is used
// ) {
//   assert(dData != nullptr);
//   assert(dMatPtr != nullptr);
//   assert(kernelInfo.cuTuple.cuContext != nullptr);
//   assert(kernelInfo.cuTuple.cuModule != nullptr);
//   assert(kernelInfo.cuTuple.cuFunction != nullptr);
//   cuCtxSetCurrent(kernelInfo.cuTuple.cuContext);

//   // Define tile size to match kernel expectations
//   unsigned nGateQubits = kernelInfo.gate->nQubits();
//   unsigned N = 1u << nGateQubits; // Size of gate matrix (2^nGateQubits)
//   unsigned TILE = std::min(256u, N);
//   unsigned combos =
//       (nQubits > nGateQubits) ? (1u << (nQubits - nGateQubits)) : 1;
//   unsigned tilesPerGate = (N + TILE - 1) / TILE;
//   unsigned gridDimX = combos * tilesPerGate;

//   void* kernelParams[] = {&dData, &dMatPtr, &combos};

//   // clang-format off
//   CU_CALL(cuLaunchKernel(kernelInfo.cuTuple.cuFunction,
//                          gridDimX, 1, 1,                    // grid dim
//                          TILE, 1, 1,                        // block dim
//     0,
//                          0,                                 // stream
//                          kernelParams,                      // kernel arguments
//                          nullptr),
//           "launchCUDAKernelParam");
//   // clang-format on
// }

#undef DEBUG_TYPE