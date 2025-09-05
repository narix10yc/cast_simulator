#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/Config.h"

#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
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
     << "LLVM Module      : " << llvmModule << "\n";
  return os << CYAN("================================\n");
}

std::ostream& CUDAKernelManager::displayInfo(std::ostream& os) const {
  os << CYAN("=== Info of CUDA Kernel Manager @ " << (void*)this << " ===\n")
     << "Num Worker Threads : " << dispatcher.getNumWorkers() << "\n"
     << "CU Device          : " << cuDevice << "\n"
     << "Primary CU Context : " << primaryCuCtx << "\n"
     << "Primary CU Stream  : " << primaryCuStream << "\n";

  int nKernels = 0;
  size_t totalPTXSize = 0, totalCUBINSize = 0;
  unsigned nActivePTX = 0, nActiveCUBIN = 0;
  for (const auto& kernel : *this) {
    ++nKernels;
    if (!kernel.ptxString.empty())
      ++nActivePTX;
    if (!kernel.cubinData.empty())
      ++nActiveCUBIN;
    totalPTXSize += kernel.ptxString.size();
    totalCUBINSize += kernel.cubinData.size();
  }
  os << "Num Kernels        : " << nKernels << "\n"
     << "PTX     : " << nActivePTX << " availables\n"
     << "  Total PTX Size   : " << utils::fmt_mem(totalPTXSize) << "\n"
     << "CUBIN   : " << nActiveCUBIN << " availables\n"
     << "  Total CUBIN Size : " << utils::fmt_mem(totalCUBINSize) << "\n";
  return os << CYAN("================================\n");
}

std::ostream&
CUDAKernelManager::ExecutionResult::displayInfo(std::ostream& os) const {
  os << "Kernel Name       : " << kernelName << "\n"
     << "PTX -> CUBIN time : "
     << utils::fmt_time(std::chrono::duration<double>(t_cubinPrepareFinish -
                                                      t_cubinPrepareStart)
                            .count())
     << "\n"
     << "Kernel Time       : " << utils::fmt_time(kernelTime_ms * 1e-3) << "\n";
  return os;
}

CUDAKernelManager::CUDAKernelManager(int nWorkerThreads, int deviceOrdinal)
    : KernelManager<CUDAKernelInfo>(
          nWorkerThreads > 0 ? nWorkerThreads : cast::get_cpu_num_threads()) {
  CU_CHECK(cuInit(0));
  CU_CHECK(cuDeviceGet(&cuDevice, deviceOrdinal));
  CU_CHECK(cuDevicePrimaryCtxRetain(&primaryCuCtx, cuDevice));
  CU_CHECK(cuCtxSetCurrent(primaryCuCtx));
  CU_CHECK(cuStreamCreate(&primaryCuStream, CU_STREAM_DEFAULT));

  // initialize exec thread
  execTh_ = std::thread(&CUDAKernelManager::execTh_work_, this);
}

CUDAKernelManager::~CUDAKernelManager() {
  syncKernelExecution();
  execStopFlag_ = true;
  execCV_.notify_one();
  assert(execTh_.joinable());
  execTh_.join();

  // destroy the primary stream
  assert(primaryCuStream != nullptr);
  assert(cuStreamSynchronize(primaryCuStream) == CUDA_SUCCESS);
  CU_CHECK(cuStreamDestroy(primaryCuStream));

  // release the primary context
  CU_CHECK(cuDevicePrimaryCtxRelease(cuDevice));
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

llvm::Error CUDAKernelManager::compileLLVMIRToPTX(int llvmOptLevel,
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
  if (target == nullptr)
    return llvm::createStringError("Failed to lookup target: " + err);

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
      return llvm::createStringError("Module verification failed: " + errorStr);
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
  return llvm::Error::success();
}

static inline llvm::OptimizationLevel wrapLLVMOptLevel(int llvmOptLevel) {
  if (llvmOptLevel == 0)
    return llvm::OptimizationLevel::O0;
  if (llvmOptLevel == 1 || llvmOptLevel < 0)
    return llvm::OptimizationLevel::O1;
  if (llvmOptLevel == 2)
    return llvm::OptimizationLevel::O2;
  // llvmOptLevel >= 3
  return llvm::OptimizationLevel::O3;
}

// TODO: Current each call to this function creates a new set of analysis
// managers. We could try to make these things thread-local for better
// performance
static void optimizeLLVMIR_work(int llvmOptLevel, CUDAKernelInfo& kernel) {
  auto optLevel = wrapLLVMOptLevel(llvmOptLevel);

  // --- Analysis managers (must be constructed in this order) ---
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // --- Pass instrumentation (debug hooks, timers, etc.) ---
  PassInstrumentationCallbacks PIC;
  StandardInstrumentations SI(*kernel.llvmContext, /*DebugLogging=*/false);
  SI.registerCallbacks(PIC, &MAM);

  // Use the PassBuilder ctor that takes PIC (portable for LLVM 20.x).
  PipelineTuningOptions PTO;
  PassBuilder PB(/*TM=*/nullptr, PTO, /*PGO=*/std::nullopt, &PIC);

  // Register analyses and cross-proxies.
  PB.registerLoopAnalyses(LAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerModuleAnalyses(MAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Wrap default pipeline with verifiers (pre & post).
  ModulePassManager MPM;
  MPM.addPass(VerifierPass()); // verify pre
  MPM.addPass(PB.buildPerModuleDefaultPipeline(optLevel));
  MPM.addPass(VerifierPass()); // verify post

  // Run the pipeline for this module.
  MPM.run(*kernel.llvmModule, MAM);
}

static llvm::Error compileLLVMIRToPTX_work(CUDAKernelInfo& kernel) {
  std::string targetTriple = "nvptx64-nvidia-cuda";
  std::string err;
  const auto* target = TargetRegistry::lookupTarget(targetTriple, err);
  if (!target) {
    return llvm::createStringError("Failed to lookup target: " + err);
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
  kernel.llvmModule->setTargetTriple(targetTriple);
  kernel.llvmModule->setDataLayout(createTargetMachine()->createDataLayout());
  std::string errorStr;
  llvm::raw_string_ostream sstream(errorStr);
  if (llvm::verifyModule(*kernel.llvmModule, &sstream))
    return llvm::createStringError("Module verification failed: " + errorStr);

  raw_pwrite_vector_ostream vecStream(kernel.ptxString);
  legacy::PassManager passManager;
  // this function returns false on success
  auto r = createTargetMachine()->addPassesToEmitFile(
      passManager, vecStream, nullptr, CodeGenFileType::AssemblyFile);
  if (r)
    return llvm::createStringError("LLVM target machine can't emit PTX");

  passManager.run(*(kernel.llvmModule));
  return llvm::Error::success();
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

llvm::Error CUDAKernelManager::compilePTXToCubin(int cuOptLevel, int verbose) {
  if (cuOptLevel < 0)
    cuOptLevel = 1;
  if (cuOptLevel > 4)
    cuOptLevel = 4;

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

  return llvm::Error::success();
}

void CUDAKernelManager::execTh_work_() {
  CU_CHECK(cuCtxSetCurrent(primaryCuCtx));

  const auto unloadLaunchTask = [this](LaunchTask& task) {
    {
      std::lock_guard lk(execMtx_);
      std::cerr << "Unloading finished kernel " << task.er->kernelName << "\n";
    }
    task.er->status.store(ExecutionResult::CleanedUp);
    assert(task.cuModule != nullptr);
    CU_CHECK(cuModuleUnload(task.cuModule));
    task.cuModule = nullptr;
    assert(task.cuFunction != nullptr);
    task.cuFunction = nullptr;
  };

  const auto loadLaunchTask = [this](LaunchTask& task, InitialLaunch& il) {
    {
      std::lock_guard lk(execMtx_);
      std::cerr << "Loading kernel " << il.er->kernelName << "\n";
    }

    il.er->status.store(ExecutionResult::Running);
    task.kernel = il.kernel;
    task.er = il.er;

    assert(task.cuModule == nullptr);
    assert(task.cuFunction == nullptr);
    assert(task.kernel->cubinData.size() > 0);

    CU_CHECK(cuModuleLoadData(&task.cuModule, task.kernel->cubinData.data()));
    CU_CHECK(cuModuleGetFunction(
        &task.cuFunction, task.cuModule, task.kernel->llvmFuncName.c_str()));

    // setup kernel launch parameters
    unsigned nCombos =
        1U << (launchConfig_.nQubits - task.kernel->gate->nQubits());
    unsigned gridDim =
        (nCombos + launchConfig_.blockSize - 1) / launchConfig_.blockSize;

    task.resetParams();
    task.addParam(launchConfig_.devicePtr);
    task.addParam(nCombos);
    auto param = task.getKernelParams();
    // clang-format off
    CU_CHECK(cuLaunchKernel(task.cuFunction,
                            gridDim, 1, 1,
                            launchConfig_.blockSize, 1, 1,
                            0, // shared mem
                            primaryCuStream,
                            param,
                            nullptr));
    // clang-format on
    task.callbackUserData.statusPtr = &task.er->status;
    task.callbackUserData.cvPtr = &execCV_;
    CU_CHECK(cuLaunchHostFunc(
        primaryCuStream,
        +[](void* userData) {
          auto* ud = static_cast<LaunchTask::CallbackUserData*>(userData);
          ud->statusPtr->store(ExecutionResult::Finished);
          std::cerr << "Kernel finished, notifying exec thread\n";
          ud->cvPtr->notify_one();
        },
        static_cast<void*>(&task.callbackUserData)));
  };

  while (true) {
    {
      std::unique_lock lk(execMtx_);
      std::cerr << "Exec thread waiting...\n";
      execCV_.wait(lk, [this] {
        // fast path: all tasks are launched and finished, notify the sync thread
        // The exec thread itself waits the execStopFlag_ to exit
        if (launchIdx == unloadedIdx && initialLaunches_.empty()) {
          syncFlag_ = true;
          std::cerr
              << "Exec thread: All kernels finished, notifying sync thread\n";
          // assumes only one thread waiting under syncCV_ (the main thread)
          syncCV_.notify_one();
          return execStopFlag_;
        }
        // can unload a finished task
        bool caseA = window_[unloadedIdx].isFinished();

        // can launch a new task
        bool caseB = (launchIdx - unloadedIdx < LaunchWindow::WS) &&
                     initialLaunches_.hasReadyToLaunch();
        return caseA || caseB || execStopFlag_;
      });
      std::cerr << "Exec thread woke up\n";
      if (execStopFlag_)
        return;
    }

    // unload finished tasks
    while (window_[unloadedIdx].isFinished()) {
      unloadLaunchTask(window_[unloadedIdx]);
      ++unloadedIdx;
    }

    // launch new tasks
    InitialLaunch il;
    while (true) {
      // not enough space in the window
      if (launchIdx - unloadedIdx >= LaunchWindow::WS)
        break;
      // no ready-to-launch tasks
      if (initialLaunches_.hasReadyToLaunch() == false)
        break;

      // launch a new task
      initialLaunches_.pop(il);
      auto& task = window_[launchIdx];
      ++launchIdx;

      loadLaunchTask(task, il);
    }
  }
}

const CUDAKernelManager::ExecutionResult*
CUDAKernelManager::enqueueKernelLaunch(CUDAKernelInfo& kernel_, int verbosity) {
  assert(launchConfig_.devicePtr != 0);
  assert(launchConfig_.blockSize > 0);
  assert(launchConfig_.nQubits > 0);

  auto* kernel = &kernel_;

  execResults_.emplace_back();
  auto* er = &execResults_.back();
  er->kernelName = kernel->llvmFuncName;

  // prepare a initial launch task
  initialLaunches_.push(kernel, er);

  // add compilation task to the dispatcher
  dispatcher.enqueue([kernel = kernel, er = er, this]() {
    CU_CHECK(cuCtxSetCurrent(primaryCuCtx));
    assert(er != nullptr);

    er->t_cubinPrepareStart = ExecutionResult::clock_t::now();

    // TODO: possible racing if the user launches the same kernel multiple
    // times.
    // Prepares cubin (if not already available)
    if (kernel->cubinData.empty()) {
      // If PTX is not available, we optimize LLVM IR and generate PTX
      if (kernel->ptxString.empty()) {
        optimizeLLVMIR_work(1, *kernel);
        llvm::cantFail(compileLLVMIRToPTX_work(*kernel));
      }
      // PTX is now available. Compile to cubin
      compileToCubin_work(1, *kernel);
    }

    er->t_cubinPrepareFinish = ExecutionResult::clock_t::now();

    // notify the exec thread that it may try to launch kernels
    er->status.store(ExecutionResult::ReadyToLaunch);
    {
      std::lock_guard lk(execMtx_);
      std::cerr << "Kernel " << er->kernelName
                << " is ready to launch, notifying exec thread\n";
    }
    // only one exec thread
    execCV_.notify_one();
  });

  execCV_.notify_one();

  return er;
}

void CUDAKernelManager::syncKernelExecution(bool progressBar) {
  // wait for all compilation tasks to finish
  // if (progressBar)
  std::cerr << "Main thread: Waiting for all compilation tasks to finish...\n";
  dispatcher.sync(progressBar);

  std::cerr << "Main thread: All compilation tasks finished, notifying exec "
               "thread\n";

  // notify exec thread so that it can launch all kernels
  execCV_.notify_one();

  {
    // if (progressBar)
    std::cerr << "Main thread: Waiting for kernel execution...\n";
    std::unique_lock lk(execMtx_);
    syncCV_.wait(lk, [this] { return syncFlag_ == true; });
    // reset for reuse
    syncFlag_ = false;
  }
  // if (progressBar)
  std::cerr << "Main thread: returned\n";
}

// void CUDAKernelManager::clearWindow_() {
//   assert(cuStreamQuery(primaryCuStream) == CUDA_SUCCESS &&
//          "CUDA stream is not idle");

//   for (auto& task : window_.tasks_) {
//     if (task.cuModule != nullptr) {
//       CU_CHECK(cuModuleUnload(task.cuModule));
//       task.cuModule = nullptr;
//     }

//     task.cuFunction = nullptr;

//     // Record kernel time
//     if (task.startEvent && task.finishEvent) {
//       assert(cuEventQuery(task.startEvent) == CUDA_SUCCESS);
//       assert(cuEventQuery(task.finishEvent) == CUDA_SUCCESS);
//       if (task.history) {
//         CU_CHECK(cuEventElapsedTime(
//             &task.history->kernelTime_ms, task.startEvent,
//             task.finishEvent));
//       }
//     }

//     if (task.startEvent != nullptr) {
//       CU_CHECK(cuEventDestroy(task.startEvent));
//       task.startEvent = nullptr;
//     }

//     if (task.finishEvent != nullptr) {
//       CU_CHECK(cuEventDestroy(task.finishEvent));
//       task.finishEvent = nullptr;
//     }

//     task.resetParams();

//     task.status.store(LaunchTask::Idle);
//   }
// }

llvm::Error CUDAKernelManager::initJIT(int optLevel, int verbose) {
  if (auto e = compileLLVMIRToPTX(optLevel, verbose)) {
    return llvm::joinErrors(
        llvm::createStringError("Failed to compile LLVM IR to PTX: "),
        std::move(e));
  }

  if (auto e = compilePTXToCubin(optLevel, verbose)) {
    return llvm::joinErrors(
        llvm::createStringError("Failed to compile PTX to CUBIN: "),
        std::move(e));
  }

  return llvm::Error::success();
}

CUDAKernelInfo*
CUDAKernelManager::getKernelByName(const std::string& llvmFuncName) {
  for (auto& kernel : *this) {
    if (kernel.llvmFuncName == llvmFuncName)
      return &kernel;
  }
  return nullptr;
}

const CUDAKernelInfo*
CUDAKernelManager::getKernelByName(const std::string& llvmFuncName) const {
  for (const auto& kernel : *this) {
    if (kernel.llvmFuncName == llvmFuncName)
      return &kernel;
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