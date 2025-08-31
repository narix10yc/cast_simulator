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

#include <cuda_runtime.h>
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
     << "LLVM Module      : " << llvmModule << "\n";
  return os << CYAN("================================\n");
}

std::ostream& CUDAKernelManager::displayInfo(std::ostream& os) const {
  os << CYAN("=== Info of CUDA Kernel Manager @ " << (void*)this << " ===\n")
     << "Num Worker Threads : " << dispatcher.getNumWorkers() << "\n"
     << "JIT State          : " << static_cast<int>(jitState) << "\n"
     << "Primary CU Context : " << primaryCuCtx << "\n"
     << "Primary CU Stream  : " << primaryCuStream << "\n";

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

MaybeError<void> CUDAKernelManager::compileLLVMIRToPTX(int llvmOptLevel,
                                                       int verbose) {
  if (llvmOptLevel < 0)
    llvmOptLevel = 1;
  if (llvmOptLevel > 3)
    llvmOptLevel = 3;
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
  }

  applyLLVMOptimization(optLevel, verbose > 0);

  std::string targetTriple = "nvptx64-nvidia-cuda";
  std::string err;
  const auto* target = TargetRegistry::lookupTarget(targetTriple, err);
  if (!target) {
    std::ostringstream oss;
    oss << "Failed to lookup target. Error trace: " << err << "\n";
    return cast::makeError(oss.str());
  }

  int major = 0, minor = 0;
  cast::getCudaComputeCapability(major, minor);
  std::ostringstream archOss;
  archOss << "sm_" << major << minor;
  std::string archString = archOss.str();

  const auto createTargetMachine = [&]() -> TargetMachine* {
    return target->createTargetMachine(
        targetTriple, archString, "", {}, std::nullopt);
  };

  // Prepare modules (DL, verify)
  for (auto& pair : llvmContextModulePairs) {
    llvm::Module* mod = pair.llvmModule.get();
    mod->setTargetTriple(targetTriple);
    mod->setDataLayout(createTargetMachine()->createDataLayout());
    std::string errorStr;
    llvm::raw_string_ostream sstream(errorStr);
    if (llvm::verifyModule(*mod, &sstream))
      return cast::makeError("Module verification failed: " + errorStr);
  }

  for (auto& kernel : *this) {
    dispatcher.enqueue([&]() {
      raw_pwrite_vector_ostream vecStream(kernel.ptxString);
      legacy::PassManager passManager;
      auto r = createTargetMachine()->addPassesToEmitFile(
          passManager, vecStream, nullptr, CodeGenFileType::AssemblyFile);
      assert(!r && "LLVM target machine can't emit a file of this type");
      passManager.run(*(kernel.llvmModule));
    });
  }

  if (verbose > 0)
    std::cerr << "Generating PTX codes...\n";
  dispatcher.sync(/* progressBar */ verbose > 0);
  jitState = JIT_CompiledPTX;
  return {}; // success
}

static void compileLLVMIRToPTX_work(int llvmOptLevel, CUDAKernelInfo& kernel) {

}

// Must call cuCtxSetCurrent before calling this function
static void compileToCubin_work(int cuOptLevel, CUDAKernelInfo& kernel) {
  assert(!kernel.ptxString.empty());

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

  CU_CHECK(cuLinkDestroy(linkState));
}

MaybeError<void> CUDAKernelManager::compilePTXToCubin(int cuOptLevel,
                                                      int verbose) {
  if (cuOptLevel < 0)
    cuOptLevel = 1;
  if (cuOptLevel > 4)
    cuOptLevel = 4;

  if (jitState != JIT_CompiledPTX) {
    return cast::makeError(
        "PTX must be available when calling compilePTXToCubin");
  }

  for (auto& kernel : *this) {
    dispatcher.enqueue([=, this, &kernel]() {
      // must set cuContext before calling compileToCubin_work
      CU_CHECK(cuCtxSetCurrent(primaryCuCtx));
      compileToCubin_work(cuOptLevel, kernel);
    });
  }

  if (verbose > 0)
    std::cerr << "JIT Compile PTX to CUBIN...\n";
  dispatcher.sync(/* progressBar */ verbose > 0);

  jitState = JIT_CompiledCubin;
  return {}; // success
}

void CUDAKernelManager::execThreadWork() {
  // Bind Driver (primary) in THIS thread
  static thread_local CUcontext bound = nullptr;
  if (bound != primaryCuCtx) {
    CU_CHECK(cuCtxSetCurrent(primaryCuCtx));
    bound = primaryCuCtx;
  }

  // Attach Runtime in THIS thread to the same device’s primary
  cudaSetDevice(0);
  cudaFree(0); // forces attach; cheap no-op allocation/free

  // CU_CHECK(cuCtxSetCurrent(primaryCuCtx));
  while (true) {
    {
      std::unique_lock<std::mutex> lk(execMtx_);
      // resumes when either
      // - there exists window seats and pending_ is not empty, or
      // - there are kernels available to be launched
      std::cerr << "Exec thread waiting...\n";
      execCV_.wait(lk, [this]() {
        // loadIdx is atomic, launchIdx and pending_ are only accessed by the
        // exec thread
        auto nWindowSeats = loadIdx - launchIdx;
        bool caseA = (nWindowSeats < Window::WS && !pending_.empty());
        bool caseB = window_[launchIdx].status.load() == LaunchTask::Ready;
        bool caseC = (stopFlag_ == true);
        return (caseA || caseB || caseC);
      });
      std::cerr << "Exec thread resuming\n";
    }
    if (stopFlag_ == true) {
      std::cerr << "Exec thread stopping\n";
      return;
    }

    // launch as many kernels as we can
    while (true) {
      auto& window = window_[launchIdx]; // launch config
      // kernel is not available. End here
      if (window.status != LaunchTask::Ready)
        break;

      // From now on we do not need locks because no loading thread will be
      // working on this until this thread (exec thread) pose another task later
      // Launch the kernel
      ++launchIdx;
      void** kernelParams = window.getKernelParams();
      if (window.startEvent == nullptr)
        CU_CHECK(cuEventCreate(&window.startEvent, CU_EVENT_DEFAULT));
      if (window.finishEvent == nullptr)
        CU_CHECK(cuEventCreate(&window.finishEvent, CU_EVENT_DEFAULT));

      CU_CHECK(cuEventRecord(window.startEvent, primaryCuStream));
      // clang-format off
      CU_CHECK(cuLaunchKernel(window.cuFunction,
                              window.gridSize, 1, 1,
                              window.blockSize, 1, 1,
                              0, /* shared memory size */
                              primaryCuStream, /* stream */
                              kernelParams,
                              nullptr));
      // clang-format on

      CU_CHECK(cuEventRecord(window.finishEvent, primaryCuStream));
      window.status.store(LaunchTask::Running);
      loadCV_.notify_all();
    }

    // pose loading requests
    assert(launchConfig_.dData);
    while (true) {
      CUDAKernelInfo* kernel;
      {
        std::unique_lock lock(execMtx_);
        if (pending_.empty())
          break;
        kernel = pending_.front();
        pending_.pop_front();
      }

      // A loading worker thread will be accessing this kernel info
      // When finishes, it marks ready as true and notify the exec thread.
      auto* window = &window_[loadIdx];
      ++loadIdx;
      dispatcher.enqueue([=, this]() {
        CU_CHECK(cuCtxSetCurrent(primaryCuCtx));
        if (kernel->cubinData.empty())
          compileToCubin_work(1, *kernel);

        {
          // When the status is Ready, we need to wait for the exec thread to
          // launch it
          std::unique_lock lock(execMtx_);
          loadCV_.wait(lock, [=] {
            auto status = window->status.load();
            return status != LaunchTask::Ready;
          });
        }
        // wait for the previous kernel to finish
        if (window->status != LaunchTask::Uninited) {
          // if the kernel is running or finished, we sync it and report the
          // execution time
          assert(window->startEvent != nullptr);
          assert(window->finishEvent != nullptr);
          CU_CHECK(cuEventSynchronize(window->startEvent));
          CU_CHECK(cuEventSynchronize(window->finishEvent));
          float ms;
          CU_CHECK(
              cuEventElapsedTime(&ms, window->startEvent, window->finishEvent));

          assert(window->cuModule != nullptr);
          CU_CHECK(cuModuleUnload(window->cuModule));
          window->cuModule = nullptr;
          window->status = LaunchTask::Finished;
        }

        assert(window->cuModule == nullptr);
        assert(kernel->cubinData.size() > 0);
        CU_CHECK(cuModuleLoadData(&window->cuModule, kernel->cubinData.data()));
        CU_CHECK(cuModuleGetFunction(&window->cuFunction,
                                     window->cuModule,
                                     kernel->llvmFuncName.c_str()));
        unsigned nCombos = 1U
                           << (launchConfig_.nQubits - kernel->gate->nQubits());
        unsigned gridDim =
            (nCombos + launchConfig_.blockSize - 1) / launchConfig_.blockSize;

        window->resetParams();
        window->addParam(launchConfig_.dData);
        window->addParam(nCombos);
        window->gridSize = gridDim;
        window->blockSize = launchConfig_.blockSize;

        window->status.store(LaunchTask::Ready);
        execCV_.notify_one();
      });
    }
  }
}
// MaybeError<void> CUDAKernelManager::loadCubin(int verbose) {
//   assert(jitState == JIT_CompiledCubin);
//   assert(nWorkerThreads_ > 0);

//   utils::TaskDispatcher dispatcher(nWorkerThreads_);

//   // this->cuModules takes the ownership of all cuda modules created in
//   // loadCubin_work
//   cuModules.resize(this->numKernels());
//   CUmodule* cuModulePtr = cuModules.data();
//   for (auto& kernel : *this) {
//     dispatcher.enqueue([=, this, &kernel]() {
//       loadCubin_work(kernel);
//       *cuModulePtr = kernel.cuModule;
//     });
//     ++cuModulePtr;
//   }

//   if (verbose > 0)
//     std::cerr << "Load CUBIN...\n";
//   dispatcher.sync(/* progressBar */ verbose > 0);
//   return {}; // success
// }

MaybeError<void> CUDAKernelManager::initJIT(int optLevel, int verbose) {
  if (jitState != JIT_Uninited) {
    return cast::makeError("The kernel manager must be in JIT-uninitialized "
                           "state when calling initJIT");
  }

  {
    auto r = compileLLVMIRToPTX(optLevel, verbose);
    if (!r)
      return cast::makeError("Failed to compile LLVM IR to PTX: " +
                             r.takeError());
  }
  {
    auto r = compilePTXToCubin(1, verbose);
    if (!r)
      return cast::makeError("Failed to compile PTX to CUBIN: " +
                             r.takeError());
  }

  return {}; // success
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
//                          kernelParams,                      // kernel
//                          arguments nullptr),
//           "launchCUDAKernelParam");
//   // clang-format on
// }

#undef DEBUG_TYPE