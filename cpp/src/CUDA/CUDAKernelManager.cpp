#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/Config.h"

#include "utils/Formats.h"
#include "utils/InfoLogger.h"

#include <atomic>
#include <iostream>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Error.h>
#include <llvm/Target/TargetMachine.h>

#include <nvJitLink.h>

#define DEBUG_TYPE "kernel-mgr-cuda"
#include <llvm/Support/Debug.h>

// #define LLVM_DEBUG(X) X

using namespace cast;
using namespace llvm;

std::atomic_size_t CudaKernel::globalIdCounter = 0;
std::atomic_size_t LaunchTask::globalCounter_ = 0;

LaunchTask* LaunchTaskHandler::get() const { return km.lookupLaunchTask(ptr); }

float LaunchTaskHandler::getKernelTimeMs() const {
  auto* task = get();
  return task ? task->kernelTimeMs : 0.0f;
}

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

void CudaKernel::displayInfo(utils::InfoLogger logger) const {
  auto status = this->status.load(std::memory_order_acquire);
  switch (status) {
  case Status::Empty:
    logger.put("Status", "Empty");
    return;
  case Pending:
    logger.put("Status", "Pending");
    return;
  case Status::Compiling:
    logger.put("Status", "Compiling");
    return;
  case Status::Failed:
    logger.put("Status", "Failed");
    return;
  case Status::Ready:
    logger.put("Status", "Ready");
    break; // continue to display more info
  }
  logger.put("Kernel Name", llvmFunc->getName().str())
      .put("Precision ", precision)
      .put("PTX Size  ", utils::fmt_mem(ptxString.size()))
      .put("Cubin Size", utils::fmt_mem(cubinData.size()));
}

void CUDAKernelManager::displayInfo(utils::InfoLogger logger) const {
  logger.put("CUDA Kernel Manager Info:");
  logger.put("Num Worker Threads", tPool.getNumWorkers());

  auto ongoings = this->ongoings.load(std::memory_order_acquire);
  if (ongoings > 0) {
    logger.put("Ongoing Kernel Executions", ongoings);
    return;
  }

  // no ongoing tasks, display full info
  logger.put("Pools", kernelPools_.size()).indent(2, [this](auto& l) {
    // default
    auto& defaultPool = getDefaultPool();
    l.put(DEFAULT_POOL_NAME, std::to_string(defaultPool.size()) + " kernels");

    // non-default
    for (const auto& [poolName, pool] : kernelPools_) {
      if (poolName == DEFAULT_POOL_NAME)
        continue;
      l.put(poolName, std::to_string(pool.size()) + " kernels");
    }
  });

  int nReadyKernels = 0;
  int nNotReadyKernels = 0;
  size_t ptxBytes = 0;
  size_t cubinBytes = 0;
  for (const auto& [poolName, pool] : kernelPools_) {
    for (const auto& kernel : pool) {
      if (kernel->status.load(std::memory_order_acquire) ==
          CudaKernel::Status::Ready) {
        nReadyKernels++;
        ptxBytes += kernel->ptxString.size();
        cubinBytes += kernel->cubinData.size();
      } else {
        nNotReadyKernels++;
      }
    }
  }
  logger.put("Kernels", [=](auto& os) {
    os << (nReadyKernels + nNotReadyKernels);
    if (nNotReadyKernels > 0)
      os << " (" << nNotReadyKernels << " not ready)";
  });
  if (nReadyKernels > 0) {
    logger.put("Total PTX Size", utils::fmt_mem(ptxBytes))
        .put("Total Cubin Size", utils::fmt_mem(cubinBytes));
  }
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

static llvm::Error
optimizeLLVMIR_work(int llvmOptLevel, CudaKernel& kernel, CUDAJitTls& jitTls) {
  assert(kernel.llvmModule != nullptr);
  auto& llvmModule = *kernel.llvmModule;

  auto optLevel = wrapLLVMOptLevel(llvmOptLevel);
  jitTls.runOnModule(llvmModule, optLevel);

  std::string errLog;
  llvm::raw_string_ostream rso(errLog);
  if (llvm::verifyModule(llvmModule, &rso))
    return llvm::createStringError("Module verification failed: " + errLog);

  return llvm::Error::success();
}

static llvm::Error compileLLVMIRToPTX_work(CudaKernel& kernel,
                                           CUDAJitTls& jitTls) {
  assert(kernel.llvmModule != nullptr);
  auto& llvmModule = *kernel.llvmModule;

  auto& TM = jitTls.getTargetMachine();

  llvmModule.setTargetTriple(TM.getTargetTriple().str());
  llvmModule.setDataLayout(TM.createDataLayout());
  std::string errorStr;
  llvm::raw_string_ostream sstream(errorStr);
  if (llvm::verifyModule(llvmModule, &sstream))
    return llvm::createStringError("Module verification failed: " + errorStr);

  raw_pwrite_vector_ostream vecStream(kernel.ptxString);
  legacy::PassManager PM;
  // this function returns false on success
  if (TM.addPassesToEmitFile(
          PM, vecStream, nullptr, CodeGenFileType::AssemblyFile)) {
    return llvm::createStringError("LLVM target machine can't emit PTX");
  }

  PM.run(llvmModule);

  return llvm::Error::success();
}

static llvm::Error compileToCubin_work(int cuOptLevel, CudaKernel& kernel) {
  if (kernel.ptxString.empty())
    return llvm::createStringError("PTX is empty");

  std::vector<const char*> options;
  int major, minor;
  cast::getCudaComputeCapability(major, minor);
  std::string archOption = "-arch=sm_" + std::to_string(10 * major + minor);
  options.push_back(archOption.c_str());

  nvJitLinkHandle handle;
  NVJITLINK_CHECK(handle,
                  nvJitLinkCreate(&handle, options.size(), options.data()));

  NVJITLINK_CHECK(handle,
                  nvJitLinkAddData(handle,
                                   NVJITLINK_INPUT_PTX,
                                   kernel.ptxString.data(),
                                   kernel.ptxString.size(),
                                   nullptr));

  NVJITLINK_CHECK(handle, nvJitLinkComplete(handle));
  size_t cubinSize = 0;
  NVJITLINK_CHECK(handle, nvJitLinkGetLinkedCubinSize(handle, &cubinSize));
  void* cubinOut = std::malloc(cubinSize);
  NVJITLINK_CHECK(handle, nvJitLinkGetLinkedCubin(handle, cubinOut));

  kernel.cubinData.assign(static_cast<uint8_t*>(cubinOut),
                          static_cast<uint8_t*>(cubinOut) + cubinSize);

  std::free(cubinOut);
  NVJITLINK_CHECK(handle, nvJitLinkDestroy(&handle));

  return llvm::Error::success();
}

void CUDAKernelManager::enqueueForCompilation(CudaKernel* kernel) {
  assert(kernel->llvmModule != nullptr);
  using Status = CudaKernel::Status;
  kernel->status.store(Status::Pending, std::memory_order_release);

  tPool.enqueueMayErr([this, kernel]() -> llvm::Error {
    assert(kernel->llvmModule != nullptr);
    auto& tls = tPool.getTLS();
    llvm::Error err = llvm::Error::success();

    kernel->setStatus(Status::Compiling);

    err =
        llvm::joinErrors(optimizeLLVMIR_work(1, *kernel, tls), std::move(err));
    err =
        llvm::joinErrors(compileLLVMIRToPTX_work(*kernel, tls), std::move(err));
    err = llvm::joinErrors(compileToCubin_work(1, *kernel), std::move(err));

    kernel->setStatus(Status::Ready);

    return err;
  });
}

void CUDAKernelManager::execThreadFunc_() {
  unsigned idx = 0;
  CU_CHECK(cuCtxSetCurrent(cuMgr.context));

  while (true) {
    LaunchTask* task = nullptr;
    {
      std::unique_lock lk(launchQueue_.mtx);
      launchQueue_.cv.wait(lk, [this]() {
        return execThreadStopFlag.load(std::memory_order_acquire) ||
               !launchQueue_.data.empty();
      });
      if (execThreadStopFlag.load(std::memory_order_acquire))
        break;

      assert(!launchQueue_.data.empty());
      task = launchQueue_.data.front();
      launchQueue_.data.pop_front();
    }
    assert(task != nullptr);
    auto* kernel = task->kernel;
    assert(kernel != nullptr);

    LLVM_DEBUG(std::cerr << "[Exec Thread] Picked up kernel "
                         << kernel->llvmFunc->getName().str() << " @ "
                         << (void*)kernel << " for launch\n";);

    // launch the kernel
    // wait for the kernel to be ready
    for (;;) {
      auto s = kernel->status.load(std::memory_order_acquire);
      if (s == CudaKernel::Status::Ready)
        break;
      kernel->status.wait(s);
      LLVM_DEBUG(std::cerr << "[Exec Thread] Waiting for kernel "
                           << kernel->llvmFunc->getName().str() << " @ "
                           << (void*)kernel << " to be ready\n";);
    }

    LLVM_DEBUG(std::cerr << "Kernel " << kernel->llvmFunc->getName().str()
                         << " @ " << (void*)kernel
                         << " is ready for launch\n";);

    // Launch the kernel into this slot
    auto& slot = launchWindow_[idx++];
    if (!slot.finished.load(std::memory_order_acquire)) {
      // wait for the launch slot to be finished
      slot.finished.wait(false);
    }
    if (slot.task != nullptr) {
      // free resources in the launch slot
      slot.resetResources();
    }

    // launch the given kernel into that slot
    slot.attachTask(task, &ongoings);
    CU_CHECK(cuModuleLoadDataEx(
        &slot.cuModule, kernel->cubinData.data(), 0, nullptr, nullptr));

    LLVM_DEBUG(std::cerr << "[Exec Thread] Loaded kernel "
                         << kernel->llvmFunc->getName().str()
                         << " with cubin of size "
                         << utils::fmt_mem(kernel->cubinData.size()) << "\n";);
    CU_CHECK(cuModuleGetFunction(&slot.cuFunction,
                                 slot.cuModule,
                                 kernel->llvmFunc->getName().str().c_str()));

    auto nCombos = 1ULL << (cuMgr.sv->nQubits() - kernel->gate->nQubits());
    slot.setArgs(cuMgr.sv->getDevicePtr(), 0, nCombos);
    slot.finished.store(false, std::memory_order_release);

    if (timingEnabled.load(std::memory_order_acquire)) {
      if (slot.startEvent == nullptr)
        CU_CHECK(cuEventCreate(&slot.startEvent, CU_EVENT_DEFAULT));
      if (slot.stopEvent == nullptr)
        CU_CHECK(cuEventCreate(&slot.stopEvent, CU_EVENT_DEFAULT));
      CU_CHECK(cuEventRecord(slot.startEvent, 0));
    }
    CU_CHECK(cuLaunchKernel(slot.cuFunction,
                            1,
                            1,
                            1, // grid dim
                            32,
                            1,
                            1, // block dim
                            0, // shared mem
                            0, // stream
                            slot.getArgs(),
                            nullptr));
    if (timingEnabled.load(std::memory_order_acquire)) {
      CU_CHECK(cuEventRecord(slot.stopEvent, 0));
    }
    LLVM_DEBUG(std::cerr << "[Exec Thread] Launched kernel "
                         << kernel->llvmFunc->getName().str() << " in slot "
                         << idx - 1 << "\n";);
    CU_CHECK(
        cuLaunchHostFunc(0,
                         CUDAKernelManager::LaunchSlot::setFinishedCallback,
                         &slot.onFinishUserData));
  } // while (true)
}

LaunchTask* CUDAKernelManager::lookupLaunchTask(LaunchTask* ptr) {
  return launchHistory_.lookup(ptr);
}

llvm::Error CUDAKernelManager::syncCompilation() {
  tPool.sync();
  return tPool.takeError();
}

llvm::Expected<LaunchTaskHandler>
CUDAKernelManager::enqueueKernelExecution(CudaKernel* kernel) {
  if (kernel == nullptr)
    return llvm::createStringError("Kernel pointer is null.");

  if (cuMgr.sv == nullptr) {
    return llvm::createStringError(
        "No statevector is attached to the kernel manager.");
  }

  auto task = std::make_unique<LaunchTask>();
  auto* taskPtr = task.get();
  task->kernel = kernel;
  launchHistory_.insert(std::move(task));
  {
    std::lock_guard lk(launchQueue_.mtx);
    launchQueue_.data.push_back(taskPtr);
    launchQueue_.cv.notify_one();
  }
  ongoings.fetch_add(1, std::memory_order_acq_rel);
  return LaunchTaskHandler(*this, taskPtr);
}

llvm::Error CUDAKernelManager::syncKernelExecution() {
  tPool.sync();
  auto err = tPool.takeError();

  // wait for all launch tasks to finish
  for (;;) {
    auto cur = ongoings.load(std::memory_order_acquire);
    if (cur == 0)
      break;
    ongoings.wait(cur, std::memory_order_acquire);
  }

  return err;
}

#undef DEBUG_TYPE
