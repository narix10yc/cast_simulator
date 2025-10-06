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
     << "Function Name    : " << getName() << "\n"
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
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

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
static llvm::Error optimizeLLVMIR_work(int llvmOptLevel,
                                       CUDAKernelInfo& kernel) {
  auto optLevel = wrapLLVMOptLevel(llvmOptLevel);

  // --- Analysis managers (must be constructed in this order) ---
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // TODO: pass an explicit TargetMachine
  PassBuilder PB(/*TM=*/nullptr);

  // Register analyses and cross-proxies.
  PB.registerLoopAnalyses(LAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerModuleAnalyses(MAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Wrap default pipeline with verifiers (pre & post).
  ModulePassManager MPM;
  MPM.addPass(PB.buildPerModuleDefaultPipeline(optLevel));

  // Run the pipeline for this module.
  MPM.run(*kernel.llvmFunc->getParent(), MAM);

  std::string errLog;
  llvm::raw_string_ostream rso(errLog);
  if (llvm::verifyModule(*kernel.llvmFunc->getParent(), &rso))
    return llvm::createStringError("Module verification failed: " + errLog);

  return llvm::Error::success();
}

// TODO: use TLS to avoid creating target machine every time
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

  auto* llvmModule = kernel.llvmFunc->getParent();
  llvmModule->setTargetTriple(targetTriple);
  llvmModule->setDataLayout(createTargetMachine()->createDataLayout());
  std::string errorStr;
  llvm::raw_string_ostream sstream(errorStr);
  if (llvm::verifyModule(*llvmModule, &sstream))
    return llvm::createStringError("Module verification failed: " + errorStr);

  raw_pwrite_vector_ostream vecStream(kernel.ptxString);
  legacy::PassManager PM;
  // this function returns false on success
  auto r = createTargetMachine()->addPassesToEmitFile(
      PM, vecStream, nullptr, CodeGenFileType::AssemblyFile);
  if (r)
    return llvm::createStringError("LLVM target machine can't emit PTX");

  PM.run(*llvmModule);

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
                         kernel.getName().data(),
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

namespace {
using ExecutionResult = CUDAKernelManager::ExecutionResult;
struct LaunchTask {
  std::vector<std::unique_ptr<std::byte[]>> kernelParamsStorage_;
  mutable std::vector<void*> kernelParams_;

  CUDAKernelInfo* kernel = nullptr;
  ExecutionResult* er = nullptr;

  CUmodule cuModule = nullptr;
  CUfunction cuFunction = nullptr;

  CUevent startEvent = nullptr;
  CUevent finishEvent = nullptr;

  struct CallbackUserData {
    decltype(ExecutionResult::status)* statusPtr;
    std::condition_variable* cvPtr;
  };

  CallbackUserData callbackUserData{};

  LaunchTask() = default;
  LaunchTask(CUDAKernelInfo* kernel, ExecutionResult* er)
      : kernel(kernel), er(er) {}

  // Does nothing if either event is nullptr or if er is nullptr
  void tryRecordTime() {
    if (er == nullptr)
      return;
    assert((startEvent == nullptr) ^ (finishEvent != nullptr));
    if (startEvent == nullptr || finishEvent == nullptr)
      return;
    assert(cuEventQuery(startEvent) == CUDA_SUCCESS);
    assert(cuEventQuery(finishEvent) == CUDA_SUCCESS);
    CU_CHECK(cuEventElapsedTime(&er->kernelTime_ms, startEvent, finishEvent));
  }

  bool isFinished() const {
    return er != nullptr && er->status.load() == ExecutionResult::Finished;
  }

  template <typename T> void addParam(T param) {
    kernelParams_.clear();
    auto paramPtr = std::make_unique<std::byte[]>(sizeof(T));
    std::memcpy(paramPtr.get(), &param, sizeof(T));
    kernelParamsStorage_.push_back(std::move(paramPtr));
  }

  void resetParams() {
    kernelParamsStorage_.clear();
    kernelParams_.clear();
  }

  // The retained pointer remains valid until the next call to addParam
  // or resetParams.
  void** getKernelParams() const {
    if (!kernelParams_.empty())
      return kernelParams_.data();

    kernelParams_.reserve(kernelParamsStorage_.size());
    for (auto& p : kernelParamsStorage_)
      kernelParams_.push_back(static_cast<void*>(p.get()));
    return kernelParams_.data();
  }
}; // struct LaunchTask

struct LaunchWindow {
  // window size
  static constexpr int WS = 4;
  std::array<LaunchTask, WS> tasks_;

  LaunchTask& operator[](int idx) { return tasks_[idx % WS]; }
}; // struct LaunchWindow

} // anonymous namespace

void CUDAKernelManager::execTh_work_() {
  // monotonically increasing indices.
  unsigned launchIdx = 0, unloadedIdx = 0;

  // When task status changes to Ready, the exec thread moves the task from
  // initialLaunches_ to launchWindow_.
  LaunchWindow window;

  CU_CHECK(cuCtxSetCurrent(primaryCuCtx));

  const auto loadLaunchTask = [&](CUDAKernelInfo* kernel, ExecutionResult* er) {
    auto& task = window[launchIdx++];
    task.kernel = kernel;
    task.er = er;
    LLVM_DEBUG({
      std::lock_guard lk(execMtx_);
      std::cerr << "+ kernel " << task.er->kernelName << ", launchIdx now "
                << launchIdx << ", " << initialLaunches_.size() << " pending\n";
    });
    assert(task.kernel != nullptr);
    assert(task.er != nullptr);

    assert(task.er->status.load() == ExecutionResult::ReadyToLaunch);

    task.er->status.store(ExecutionResult::Running);
    assert(task.cuModule == nullptr);
    assert(task.cuFunction == nullptr);
    {
      std::lock_guard lk(execMtx_);
      assert(task.kernel->cubinData.size() > 0);
      CU_CHECK(cuModuleLoadData(&task.cuModule, task.kernel->cubinData.data()));
    }
    CU_CHECK(cuModuleGetFunction(
        &task.cuFunction, task.cuModule, task.kernel->getName().data()));

    // setup kernel launch parameters
    unsigned nCombos =
        1U << (launchConfig_.nQubitsSV - task.kernel->gate->nQubits());
    unsigned gridDim =
        (nCombos + launchConfig_.blockSize - 1) / launchConfig_.blockSize;

    task.addParam(launchConfig_.devicePtr);
    task.addParam(nCombos);
    auto param = task.getKernelParams();

    // setup timing if enabled
    if (this->timingEnabled_) {
      if (task.startEvent == nullptr)
        CU_CHECK(cuEventCreate(&task.startEvent, CU_EVENT_DEFAULT));
      if (task.finishEvent == nullptr)
        CU_CHECK(cuEventCreate(&task.finishEvent, CU_EVENT_DEFAULT));
    }

    if (this->timingEnabled_)
      CU_CHECK(cuEventRecord(task.startEvent, primaryCuStream));
    // clang-format off
    CU_CHECK(cuLaunchKernel(task.cuFunction,
                            gridDim, 1, 1,
                            launchConfig_.blockSize, 1, 1,
                            0, // shared mem
                            primaryCuStream,
                            param,
                            nullptr));
    // clang-format on
    if (this->timingEnabled_)
      CU_CHECK(cuEventRecord(task.finishEvent, primaryCuStream));

    // setup callback to mark the task as finished
    task.callbackUserData.statusPtr = &task.er->status;
    task.callbackUserData.cvPtr = &execCV_;
    CU_CHECK(cuLaunchHostFunc(
        primaryCuStream,
        +[](void* userData) {
          auto* ud = static_cast<LaunchTask::CallbackUserData*>(userData);
          ud->statusPtr->store(ExecutionResult::Finished);
          ud->cvPtr->notify_one();
        },
        static_cast<void*>(&task.callbackUserData)));
  }; // lambda loadLaunchTask

  while (true) {
    {
      std::unique_lock lk(execMtx_);
      execCV_.wait(lk, [&] {
        // fast path: syncing requested by the main thread and all tasks are
        // launched and finished, notify the main thread.
        // The exec thread itself checks the execStopFlag_ to exit
        if (syncFlag_ == RequestSyncing && launchIdx == unloadedIdx &&
            initialLaunches_.empty()) {
          // std::cerr << "= all kernels finished. Notify syncing threads\n";
          syncFlag_ = Synced;
          // assumes only one thread waiting under syncCV_ (the main thread)
          syncCV_.notify_one();
          return execStopFlag_;
        }
        // can unload a finished task
        bool caseA = window[unloadedIdx].isFinished();

        // can launch a new task
        bool caseB = (launchIdx - unloadedIdx < LaunchWindow::WS) &&
                     initialLaunches_.hasReadyToLaunch();
        return caseA || caseB || execStopFlag_;
      });
      if (execStopFlag_)
        return;
    }

    // unload finished tasks
    while (true) {
      auto& task = window[unloadedIdx];
      if (task.isFinished() == false)
        break;
      ++unloadedIdx;
      LLVM_DEBUG({
        std::lock_guard lk(execMtx_);
        std::cerr << "- kernel " << task.er->kernelName << ", unloadedIdx now "
                  << unloadedIdx << "\n";
      });
      assert(task.er != nullptr);
      assert(task.er->status.load() == ExecutionResult::Finished);

      task.er->status.store(ExecutionResult::CleanedUp);
      task.tryRecordTime();
      task.resetParams();
      assert(task.cuModule != nullptr);
      CU_CHECK(cuModuleUnload(task.cuModule));
      task.cuModule = nullptr;
      assert(task.cuFunction != nullptr);
      task.cuFunction = nullptr;
      task.kernel = nullptr;
      task.er = nullptr;
    }

    // launch new tasks

    while (true) {
      // not enough space in the window
      if (launchIdx - unloadedIdx >= LaunchWindow::WS)
        break;
      // no ready-to-launch tasks
      if (initialLaunches_.hasReadyToLaunch() == false)
        break;

      // launch a new task
      CUDAKernelInfo* ilKernel;
      ExecutionResult* ilEr;
      initialLaunches_.pop(ilKernel, ilEr);
      loadLaunchTask(ilKernel, ilEr);
    }
  }
}

const CUDAKernelManager::ExecutionResult*
CUDAKernelManager::enqueueKernelLaunch(CUDAKernelInfo& kernel_) {
  assert(launchConfig_.devicePtr != 0);
  assert(launchConfig_.blockSize > 0);
  assert(launchConfig_.nQubitsSV > 0);

  auto* kernel = &kernel_;

  execResults_.emplace_back();
  auto* er = &execResults_.back();
  er->kernelName = kernel->getName();

  // prepare a initial launch task
  auto* semaphore = initialLaunches_.push(kernel, er);

  // add compilation task to the dispatcher
  dispatcher.enqueue([kernel, er, semaphore, this]() {
    CU_CHECK(cuCtxSetCurrent(primaryCuCtx));
    assert(semaphore != nullptr);

    er->t_cubinPrepareStart = ExecutionResult::clock_t::now();
    {
      std::unique_lock lk(semaphore->mtx);
      auto ilStatus = semaphore->status;
      if (ilStatus == KernelSemaphore::Pending) {
        // Prepares cubin (if not already available)
        semaphore->status = KernelSemaphore::Compiling;
        LLVM_DEBUG({
          std::lock_guard lk(execMtx_);
          std::cerr << "Loading thread " << dispatcher.getWorkerID()
                    << " is preparing kernel " << kernel->getName() << "\n";
        });
        // unlock while doing the compilation
        lk.unlock();
        assert(kernel != nullptr);
        if (kernel->cubinData.empty()) {
          // If PTX is not available, we optimize LLVM IR and generate PTX
          if (kernel->ptxString.empty()) {
            if (auto e = optimizeLLVMIR_work(1, *kernel)) {
              std::cerr << "Failed to optimize LLVM IR for kernel "
                        << kernel->getName() << ": "
                        << llvm::toString(std::move(e)) << "\n";
              std::abort();
            }
            llvm::cantFail(compileLLVMIRToPTX_work(*kernel));
          }
          // PTX is now available. Compile to cubin
          compileToCubin_work(1, *kernel);
        }
        lk.lock();
        // finished compilation, notify all waiting threads
        semaphore->status = KernelSemaphore::Prepared;
        semaphore->cv.notify_all();
      } else if (ilStatus == KernelSemaphore::Compiling) {
        // some other thread is compiling the same kernel
        LLVM_DEBUG({
          std::lock_guard lk(execMtx_);
          std::cerr << "Loading thread " << dispatcher.getWorkerID()
                    << " is waiting for kernel " << kernel->getName()
                    << " to be prepared\n";
        });
        semaphore->cv.wait(lk, [semaphore] {
          return semaphore->status == KernelSemaphore::Prepared;
        });
      }
      assert(semaphore->status == KernelSemaphore::Prepared);
    }

    er->t_cubinPrepareFinish = ExecutionResult::clock_t::now();

    // notify the exec thread that it may try to launch kernels
    er->status.store(ExecutionResult::ReadyToLaunch);
    // only one exec thread
    execCV_.notify_one();
  });

  execCV_.notify_one();

  return er;
}

void CUDAKernelManager::syncKernelExecution(bool progressBar) {
  // blocks until all compilation tasks finish
  if (progressBar)
    std::cerr << "Waiting for all compilation tasks to finish...\n";
  dispatcher.sync(progressBar);

  {
    std::unique_lock lk(execMtx_);
    if (progressBar)
      std::cerr << "Waiting for kernel execution...\n";
    syncFlag_ = RequestSyncing;
    // notify exec thread so that it can launch all kernels
    execCV_.notify_one();
    syncCV_.wait(lk, [this] { return syncFlag_ == Synced; });
    // reset for reuse
    syncFlag_ = NotSyncing;
  }
}

CUDAKernelInfo*
CUDAKernelManager::getKernelByName(const std::string& llvmFuncName) {
  for (auto& kernel : *this) {
    if (kernel.getName() == llvmFuncName)
      return &kernel;
  }
  return nullptr;
}

const CUDAKernelInfo*
CUDAKernelManager::getKernelByName(const std::string& llvmFuncName) const {
  for (const auto& kernel : *this) {
    if (kernel.getName() == llvmFuncName)
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

#undef DEBUG_TYPE