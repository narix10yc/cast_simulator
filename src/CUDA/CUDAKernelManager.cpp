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
  window_ = LaunchWindow(this->dispatcher.getNumWorkers());
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
  stopFlag_ = true;
  execCV_.notify_one();
  assert(execTh_.joinable());
  execTh_.join();
  // manually unload CUDA resources
  assert(primaryCuStream != nullptr);
  CU_CHECK(cuStreamSynchronize(primaryCuStream));
  CU_CHECK(cuStreamDestroy(primaryCuStream));

  for (auto& task : window_.tasks_) {
    if (task.startEvent)
      CU_CHECK(cuEventDestroy(task.startEvent));
    if (task.finishEvent)
      CU_CHECK(cuEventDestroy(task.finishEvent));
    if (task.cuModule)
      CU_CHECK(cuModuleUnload(task.cuModule));
  }

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

  while (true) {
    {
      std::unique_lock<std::mutex> lk(execMtx_);
      execCV_.wait(lk, [this]() {
        // caseB: we have a kernel ready to launch
        bool caseB = window_[launchIdx].status.load() == LaunchTask::Ready;
        // caseC: stop flag is set
        bool caseC = (stopFlag_ == true);
        return (caseB || caseC);
      });
    }
    if (stopFlag_ == true) {
      return;
    }

    // launch as many kernels as we can
    int nLaunched = 0;
    while (true) {
      auto& task = window_[launchIdx];
      // task is not ready to launch. End here
      if (task.status != LaunchTask::Ready)
        break;

      // task is ready to launch. Launch it
      ++launchIdx;
      void** kernelParams = task.getKernelParams();
      if (task.startEvent == nullptr)
        CU_CHECK(cuEventCreate(&task.startEvent, CU_EVENT_DEFAULT));
      if (task.finishEvent == nullptr)
        CU_CHECK(cuEventCreate(&task.finishEvent, CU_EVENT_DEFAULT));

      CU_CHECK(cuEventRecord(task.startEvent, primaryCuStream));
      // clang-format off
      CU_CHECK(cuLaunchKernel(task.cuFunction,
                              task.gridSize, 1, 1,
                              task.blockSize, 1, 1,
                              0, /* shared memory size */
                              primaryCuStream, /* stream */
                              kernelParams,
                              nullptr));
      // clang-format on
      CU_CHECK(cuEventRecord(task.finishEvent, primaryCuStream));
      ++nLaunched;

      // {
      //   std::unique_lock lock(execMtx_);
      //   std::cerr << "Exec thread launched cuFunction " << task.cuFunction
      //             << "\n";
      // }

      // mark task status as 'Running' and notify loading threads that they may
      // start loading subsequent kernels
      task.status.store(LaunchTask::Running);
    }

    if (nLaunched > 0)
      loadCV_.notify_all();
    // We always notify syncCV_ to update progress bars
    syncCV_.notify_all();

  } // while (true)
}

const CUDAKernelManager::ExecutionResult*
CUDAKernelManager::enqueueKernelLaunch(CUDAKernelInfo& kernel_, int verbosity) {
  assert(launchConfig_.devicePtr != 0);
  assert(launchConfig_.blockSize > 0);
  assert(launchConfig_.nQubits > 0);

  // A loading worker thread will be accessing this kernel info
  // When finishes, it marks the status as Ready and notify the exec thread.
  auto* task = &window_[loadIdx];
  ++loadIdx;
  auto* kernel = &kernel_;

  ExecutionResult* history = nullptr;
  if (verbosity >= 1) {
    execResults_.emplace_back();
    history = &execResults_.back();
    history->kernelName = kernel->llvmFuncName;
  }

  dispatcher.enqueue([kernel = kernel, task = task, history = history, this]() {
    CU_CHECK(cuCtxSetCurrent(primaryCuCtx));

    if (history)
      history->t_cubinPrepareStart = ExecutionResult::clock_t::now();

    // TODO: possible racing if the user launches the same kernel multiple times
    // Prepares cubin (if not already available)
    if (kernel->cubinData.empty()) {
      // If PTX is not available, we optimize LLVM IR and generate PTX
      if (kernel->ptxString.empty()) {
        optimizeLLVMIR_work(1, *kernel);
        compileLLVMIRToPTX_work(*kernel);
      }
      // PTX is now available. Compile to cubin
      compileToCubin_work(1, *kernel);
    }

    if (history)
      history->t_cubinPrepareFinish = ExecutionResult::clock_t::now();

    // The loading thread continues if it finds its allocated task window is
    // Idle or Running 'Compiling'. Otherwise, if the task window is
    // - Compiling: some other loading thread is accessing it.
    // - Ready: waiting for the execution thread to launch it.
    LaunchTask::Status taskStatus;
    {
      std::unique_lock lock(execMtx_);
      loadCV_.wait(lock, [task = task, &taskStatus] {
        taskStatus = task->status.load();
        return taskStatus == LaunchTask::Idle ||
               taskStatus == LaunchTask::Running;
      });
      // Set status to Compiling so that other loading threads will wait
      task->status.store(LaunchTask::Compiling);
    }

    // wait for the previous kernel to finish
    if (taskStatus == LaunchTask::Running) {
      assert(task->finishEvent != nullptr);
      CU_CHECK(cuEventSynchronize(task->finishEvent));
      assert(cuEventQuery(task->finishEvent) == CUDA_SUCCESS);

      if (task->history) {
        assert(task->startEvent != nullptr);
        assert(cuEventQuery(task->startEvent) == CUDA_SUCCESS);

        float ms;
        CU_CHECK(cuEventElapsedTime(&ms, task->startEvent, task->finishEvent));
        task->history->kernelTime_ms = ms;
      }

      assert(task->cuModule != nullptr);
      CU_CHECK(cuModuleUnload(task->cuModule));
      task->cuModule = nullptr;
    }

    assert(task->cuModule == nullptr);
    assert(kernel->cubinData.size() > 0);
    CU_CHECK(cuModuleLoadData(&task->cuModule, kernel->cubinData.data()));
    CU_CHECK(cuModuleGetFunction(
        &task->cuFunction, task->cuModule, kernel->llvmFuncName.c_str()));
    unsigned nCombos = 1U << (launchConfig_.nQubits - kernel->gate->nQubits());
    unsigned gridDim =
        (nCombos + launchConfig_.blockSize - 1) / launchConfig_.blockSize;

    task->resetParams();
    task->addParam(launchConfig_.devicePtr);
    task->addParam(nCombos);
    task->gridSize = gridDim;
    task->blockSize = launchConfig_.blockSize;
    task->history = history;

    // {
    //   std::unique_lock lock(execMtx_);
    //   std::cerr << "Loading thread " << dispatcher.getWorkerID()
    //             << " prepared kernel " << kernel->llvmFuncName
    //             << "\n - cuFunction: " << task->cuFunction << "\n";
    // }

    // mark the task as 'Ready' and notify the exec thread that it may try to
    // launch kernels
    task->status.store(LaunchTask::Ready);
    execCV_.notify_one();
  });

  return history;
}

void CUDAKernelManager::syncKernelExecution(bool progressBar) {
  execCV_.notify_one();
  if (progressBar)
    std::cerr << "JIT compile progress:\n";
  {
    std::unique_lock lock(execMtx_);
    syncCV_.wait(lock, [=, this] {
      if (progressBar)
        utils::displayProgressBar(launchIdx, loadIdx, 20);
      return launchIdx == loadIdx;
    });
  }
  if (progressBar) {
    std::cerr << "JIT compile finished. Synchronizing CUDA stream...\n";
  }
  CU_CHECK(cuStreamSynchronize(primaryCuStream));
  clearWindow_();
}

void CUDAKernelManager::clearWindow_() {
  assert(cuStreamQuery(primaryCuStream) == CUDA_SUCCESS &&
         "CUDA stream is not idle");

  for (auto& task : window_.tasks_) {
    if (task.cuModule != nullptr) {
      CU_CHECK(cuModuleUnload(task.cuModule));
      task.cuModule = nullptr;
    }

    task.cuFunction = nullptr;

    // Record kernel time
    if (task.startEvent && task.finishEvent) {
      assert(cuEventQuery(task.startEvent) == CUDA_SUCCESS);
      assert(cuEventQuery(task.finishEvent) == CUDA_SUCCESS);
      if (task.history) {
        CU_CHECK(cuEventElapsedTime(
            &task.history->kernelTime_ms, task.startEvent, task.finishEvent));
      }
    }

    if (task.startEvent != nullptr) {
      CU_CHECK(cuEventDestroy(task.startEvent));
      task.startEvent = nullptr;
    }

    if (task.finishEvent != nullptr) {
      CU_CHECK(cuEventDestroy(task.finishEvent));
      task.finishEvent = nullptr;
    }

    task.resetParams();

    task.status.store(LaunchTask::Idle);
  }
}

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