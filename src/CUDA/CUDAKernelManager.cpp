#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/Config.h"

#include "utils/Formats.h"
#include "utils/InfoLogger.h"
#include "utils/iocolor.h"

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>

#define DEBUG_TYPE "kernel-mgr-cuda"
#include <llvm/Support/Debug.h>

// #define LLVM_DEBUG(X) X

using namespace cast;
using namespace llvm;

void CUDAKernelGenConfig::displayInfo(utils::InfoLogger logger) const {
  logger.put("Precision", precision)
      .put("Zero Tolerance", zeroTol)
      .put("One Tolerance", oneTol)
      .put("Matrix Load Mode", [&](std::ostream& os) {
        switch (matrixLoadMode) {
        case CUDAMatrixLoadMode::UseMatImmValues:
          os << "UseMatImmValues";
          break;
        case CUDAMatrixLoadMode::LoadInDefaultMemSpace:
          os << "LoadInDefaultMemSpace";
          break;
        case CUDAMatrixLoadMode::LoadInConstMemSpace:
          os << "LoadInConstMemSpace";
          break;
        default:
          os << "Unknown";
          break;
        }
      });
}

void CUDAKernelInfo::displayInfo(utils::InfoLogger logger) const {
  logger.put("Function Name", getName())
      .put("Precision", static_cast<int>(precision))
      .put("Gate Ptr", (void*)(gate.get()))
      .put("PTX",
           [&](std::ostream& os) -> void {
             if (ptxString.empty())
               os << "None";
             else
               os << "Yes, with size " << utils::fmt_mem(ptxString.size());
           })
      .put("CUBIN", [&](std::ostream& os) -> void {
        if (cubinData.empty())
          os << "None";
        else
          os << "Yes, with size " << utils::fmt_mem(cubinData.size());
      });
}

void CUDAKernelManager::displayInfo(utils::InfoLogger logger) const {
  logger.put("Num Worker Threads", tPool.getNumWorkers())
      .put("CU Device         ", cuDevice)
      .put("Primary CU Context", primaryCuCtx)
      .put("Primary CU Stream ", primaryCuStream);

  int nKernels = 0;
  size_t totalPTXSize = 0, totalCUBINSize = 0;
  unsigned nActivePTX = 0, nActiveCUBIN = 0;
  for (const auto& kernel : all_kernels()) {
    ++nKernels;
    if (!kernel->ptxString.empty())
      ++nActivePTX;
    if (!kernel->cubinData.empty())
      ++nActiveCUBIN;
    totalPTXSize += kernel->ptxString.size();
    totalCUBINSize += kernel->cubinData.size();
  }

  logger.put("Num Kernels ", nKernels)
      .put("PTX  ",
           [&](std::ostream& os) {
             if (nActivePTX == 0)
               os << "None";
             else
               os << nActivePTX << " availables, total size "
                  << utils::fmt_mem(totalPTXSize);
           })
      .put("CUBIN", [&](std::ostream& os) {
        if (nActiveCUBIN == 0)
          os << "None";
        else
          os << nActiveCUBIN << " availables, total size "
             << utils::fmt_mem(totalCUBINSize);
      });
}

void CUDAKernelManager::ExecutionResult::displayInfo(
    utils::InfoLogger logger) const {
  logger.put("Kernel Name", kernelName)
      .put("Status",
           [&](std::ostream& os) {
             switch (status.load()) {
             case Status::Pending:
               os << "Pending";
               break;
             case Status::ReadyToLaunch:
               os << "ReadyToLaunch";
               break;
             case Status::Running:
               os << "Running";
               break;
             case Status::Finished:
               os << "Finished";
               break;
             case Status::CleanedUp:
               os << "CleanedUp";
               break;
             }
           })
      .put("Compile Time (s)", utils::fmt_time(getCompileTime()))
      .put("Kernel Time (s)", [&](std::ostream& os) {
        auto t = getKernelTime();
        if (t < 0.0f)
          os << "N/A";
        else
          os << utils::fmt_time(t);
      });
}

void CUDAKernelManager::init() {
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

  CU_CHECK(cuInit(0));
  // Always use device 0 for now
  CU_CHECK(cuDeviceGet(&cuDevice, 0));
  CU_CHECK(cuDevicePrimaryCtxRetain(&primaryCuCtx, cuDevice));
  CU_CHECK(cuCtxSetCurrent(primaryCuCtx));
  CU_CHECK(cuStreamCreate(&primaryCuStream, CU_STREAM_DEFAULT));

  // Enable timing by default
  enableTiming(true);

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

// TODO: Currently each call to this function creates a new set of analysis
// managers. We could try to make these things thread-local for better
// performance
static llvm::Error optimizeLLVMIR_work(int llvmOptLevel,
                                       CUDAKernelInfo& kernel,
                                       CUDAJitTls& jitTls) {
  auto optLevel = wrapLLVMOptLevel(llvmOptLevel);
  auto* llvmModule = kernel.llvmFunc->getParent();
  jitTls.runOnModule(*llvmModule, optLevel);

  std::string errLog;
  llvm::raw_string_ostream rso(errLog);
  if (llvm::verifyModule(*llvmModule, &rso))
    return llvm::createStringError("Module verification failed: " + errLog);

  return llvm::Error::success();
}

// TODO: use TLS to avoid creating target machine every time
static llvm::Error compileLLVMIRToPTX_work(CUDAKernelInfo& kernel,
                                           CUDAJitTls& jitTls) {
  auto& TM = jitTls.getTargetMachine();

  auto* llvmModule = kernel.llvmFunc->getParent();
  llvmModule->setTargetTriple(TM.getTargetTriple());
  llvmModule->setDataLayout(TM.createDataLayout());
  std::string errorStr;
  llvm::raw_string_ostream sstream(errorStr);
  if (llvm::verifyModule(*llvmModule, &sstream))
    return llvm::createStringError("Module verification failed: " + errorStr);

  raw_pwrite_vector_ostream vecStream(kernel.ptxString);
  legacy::PassManager PM;
  // this function returns false on success
  auto r = TM.addPassesToEmitFile(
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
    std::string kernelName(task.kernel->getName());
    CU_CHECK(cuModuleGetFunction(
        &task.cuFunction, task.cuModule, kernelName.c_str()));

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
  assert(isLaunchConfigValid());

  auto* kernel = &kernel_;

  execResults_.emplace_back();
  auto* er = &execResults_.back();
  er->kernelName = kernel->getName();

  // prepare a initial launch task
  auto* semaphore = initialLaunches_.push(kernel, er);

  // add compilation task to the dispatcher
  tPool.enqueueMayErr([kernel, er, semaphore, this]() -> llvm::Error {
    CU_CHECK(cuCtxSetCurrent(primaryCuCtx));
    assert(semaphore != nullptr);

    er->t_cubinPrepareStart = clock_t::now();
    {
      std::unique_lock lk(semaphore->mtx);
      auto ilStatus = semaphore->status;
      if (ilStatus == KernelSemaphore::Pending) {
        // Prepares cubin (if not already available)
        semaphore->status = KernelSemaphore::Compiling;
        LLVM_DEBUG({
          std::lock_guard lk(execMtx_);
          std::cerr << "Loading thread " << tPool.getWorkerID()
                    << " is preparing kernel " << kernel->getName() << "\n";
        });
        // unlock while doing the compilation
        lk.unlock();
        assert(kernel != nullptr);
        if (kernel->cubinData.empty()) {
          auto& jitTls = tPool.getTLS();
          // If PTX is not available, we optimize LLVM IR and generate PTX
          if (kernel->ptxString.empty()) {
            auto e = optimizeLLVMIR_work(1, *kernel, jitTls);
            e = llvm::joinErrors(std::move(e),
                                 compileLLVMIRToPTX_work(*kernel, jitTls));
            if (e) {
              return llvm::joinErrors(
                  llvm::createStringError(
                      llvm::Twine("Failed to prepare PTX for kernel ") +
                      kernel->getName()),
                  std::move(e));
            }
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
          std::cerr << "Loading thread " << tPool.getWorkerID()
                    << " is waiting for kernel " << kernel->getName()
                    << " to be prepared\n";
        });
        semaphore->cv.wait(lk, [semaphore] {
          return semaphore->status == KernelSemaphore::Prepared;
        });
      }
      assert(semaphore->status == KernelSemaphore::Prepared);
    }

    er->t_cubinPrepareFinish = clock_t::now();

    // notify the exec thread that it may try to launch kernels
    er->status.store(ExecutionResult::ReadyToLaunch);
    // only one exec thread
    execCV_.notify_one();
    return llvm::Error::success();
  });

  execCV_.notify_one();
  return er;
}

void CUDAKernelManager::syncKernelExecution(bool progressBar) {
  // blocks until all compilation tasks finish
  if (progressBar)
    std::cerr << "Waiting for all compilation tasks to finish...\n";
  tPool.sync(progressBar);

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
  for (auto& kernel : all_kernels()) {
    if (kernel->getName() == llvmFuncName)
      return kernel.get();
  }
  return nullptr;
}

const CUDAKernelInfo*
CUDAKernelManager::getKernelByName(const std::string& llvmFuncName) const {
  for (auto& kernel : all_kernels()) {
    if (kernel->getName() == llvmFuncName)
      return kernel.get();
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