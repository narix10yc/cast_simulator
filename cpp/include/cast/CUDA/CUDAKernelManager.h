/*
 * CUDAKernelManager.h
 */
#ifndef CAST_CUDA_CUDAKERNELMANAGER_H
#define CAST_CUDA_CUDAKERNELMANAGER_H

#include "cast/Core/IRNode.h"
#include "cast/Core/QuantumGate.h"

#include "cast/CPU/Config.h" // for get_cpu_num_threads

#include "cast/CUDA/CUDAJitTls.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "cast/CUDA/Config.h"

#include "utils/InfoLogger.h"
#include "utils/ThreadPool.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>

#include <map>

#include <cuda.h>

namespace cast {

class CUDAKernelManager;
class LaunchTaskHandler;

enum class CUDAMatrixLoadMode {
  UseMatImmValues,
  LoadInDefaultMemSpace,
  LoadInConstMemSpace
};

struct CudaKernel {
  static std::atomic_size_t globalIdCounter;
  size_t id;

  // LLVMContext is not thread-safe -- for ease of parallel JIT compilation,
  // each kernel has its own LLVM context and module. Each module contains only
  // one function.
  std::unique_ptr<llvm::LLVMContext> llvmContext = nullptr;
  std::unique_ptr<llvm::Module> llvmModule = nullptr;
  llvm::Function* llvmFunc = nullptr;

  // necessary info about this kernel
  ConstQuantumGatePtr gate = nullptr;
  Precision precision = Precision::Unknown;

  // JIT session data
  std::string ptxString;
  std::vector<uint8_t> cubinData;

  // status for lock-free access
  enum Status {
    // not yet generated
    Empty,
    // enqueued for compilation
    Pending,
    // being compiled
    Compiling,
    // ready to be launched
    Ready,
    // failures
    Failed,
  };
  std::atomic<Status> status = Status::Empty;

  /// Note: the caller must ensure the uniqueness of name in the pool
  explicit CudaKernel(const std::string& name) {
    id = globalIdCounter.fetch_add(1, std::memory_order_relaxed);
    llvmContext = std::make_unique<llvm::LLVMContext>();
    llvmModule = std::make_unique<llvm::Module>(name + "_module", *llvmContext);

    assert(llvmModule != nullptr);
  }

  void setStatus(Status s) {
    status.store(s, std::memory_order_release);
    status.notify_all();
  }

  void displayInfo(utils::InfoLogger logger) const;

}; // struct CUDAKernelInfo

struct LaunchTask {
  static std::atomic_size_t globalCounter_;
  size_t id;

  CudaKernel* kernel = nullptr;
  float kernelTimeMs = 0.0f;

  LaunchTask() { id = globalCounter_.fetch_add(1, std::memory_order_relaxed); }
};

class LaunchTaskHandler {
  CUDAKernelManager& km;
  LaunchTask* ptr;

  LaunchTask* get() const;

public:
  explicit LaunchTaskHandler(CUDAKernelManager& km, LaunchTask* ptr)
      : km(km), ptr(ptr) {}

  float getKernelTimeMs() const;

  void displayInfo(utils::InfoLogger logger) const;
};

struct CUDAKernelGenConfig {
  Precision precision;
  double zeroTol = 1e-8;
  double oneTol = 1e-8;
  CUDAMatrixLoadMode matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;

  explicit CUDAKernelGenConfig(Precision p = Precision::FP64) : precision(p) {}

  void displayInfo(utils::InfoLogger logger) const;
};

/// Manages cuda-related stuff: devices, contexts, streams, modules, etc.
struct CudaCtxManager {
  std::unique_ptr<CUDAStatevectorBase> sv;

  CUdevice device;
  CUcontext context;

  CudaCtxManager() : sv() {
    CU_CHECK(cuInit(0));
    // for now always use device with ordinal 0
    CU_CHECK(cuDeviceGet(&device, 0));
    CU_CHECK(cuDevicePrimaryCtxRetain(&context, device));

    CU_CHECK(cuCtxSetCurrent(context));
  }

  CudaCtxManager(const CudaCtxManager&) = delete;
  CudaCtxManager(CudaCtxManager&&) = delete;
  CudaCtxManager& operator=(const CudaCtxManager&) = delete;
  CudaCtxManager& operator=(CudaCtxManager&&) = delete;

  ~CudaCtxManager() { CU_CHECK(cuDevicePrimaryCtxRelease(device)); };
};

class CUDAKernelManager {
  using clock_t = std::chrono::steady_clock;
  using time_point_t = clock_t::time_point;

private:
  friend class LaunchTaskHandler;
  utils::ThreadPool<CUDAJitTls> tPool;
  CudaCtxManager cuMgr;

  using Pool = std::vector<std::unique_ptr<CudaKernel>>;

  std::map<std::string, Pool> kernelPools_;
  constexpr static const char* DEFAULT_POOL_NAME = "_default_";

  void init() {
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    kernelPools_.insert({DEFAULT_POOL_NAME, Pool()});
    startExecThread_();
  }

  // Both gate-sv and superop-dm simulation will boil down to a matrix with
  // target qubits. This function contains the core logics to emit LLVM IR.
  llvm::Expected<llvm::Function*>
  gen_(const CUDAKernelGenConfig& config,
       const ComplexSquareMatrix& matrix,
       const QuantumGate::TargetQubitsType& qubits,
       const std::string& funcName,
       llvm::Module& llvmModule);

  /// An internal function to generate a CUDA kernel and put the kernel in the
  /// specified pool. This function wraps whether gate is a
  /// StandardQuantumGate (with or without noise) or SuperopQuantumGate, and
  /// call `gen_` with a corresponding ComplexSquareMatrix. The generated
  /// kernel will be put into the given pool.
  /// @param funcName: must be unique in the pool as should be guaranteed by
  /// the caller.
  /// @return: the generated CudaKernel.
  llvm::Expected<CudaKernel*> genCUDAGate_(const CUDAKernelGenConfig& config,
                                           ConstQuantumGatePtr gate,
                                           const std::string& funcName,
                                           Pool& pool);

  void enqueueForCompilation(CudaKernel* kernel);

public:
  CUDAKernelManager() : tPool(cast::get_cpu_num_threads()) { init(); }
  CUDAKernelManager(int nWorkerThreads) : tPool(nWorkerThreads) { init(); }

  CUDAKernelManager(const CUDAKernelManager&) = delete;
  CUDAKernelManager(CUDAKernelManager&&) = delete;
  CUDAKernelManager& operator=(const CUDAKernelManager&) = delete;
  CUDAKernelManager& operator=(CUDAKernelManager&&) = delete;

  ~CUDAKernelManager() { stopExecThread_(); }

  void displayInfo(utils::InfoLogger logger) const;

  Pool& getDefaultPool() { return kernelPools_.at(DEFAULT_POOL_NAME); }
  const Pool& getDefaultPool() const {
    return kernelPools_.at(DEFAULT_POOL_NAME);
  }

  /// Generate a kernel for a single gate into the default kernel pool.
  /// @param funcName: if empty, a default name "k_<index>" will be assigned.
  /// If provided, it must be unique among all kernels in the default pool.
  /// Unlike `CPUKernelManager`, here the generated kernel will be enqueued
  /// into JIT compilation session. So the returned kernel handler may not be
  /// ready for inspection immediately.
  llvm::Expected<CudaKernel*> genGate(const CUDAKernelGenConfig& config,
                                      ConstQuantumGatePtr gate,
                                      const std::string& funcName = "");

  /// Generate kernels for all gates in the given circuit graph. The generated
  /// kernels will be named as <graphName>_<order>_<gateId>, where order is
  /// the order of the gate in the circuit graph.
  // TODO: do we still need the order
  llvm::Error genGraphGates(const CUDAKernelGenConfig& config,
                            const ir::CircuitGraphNode& graph,
                            const std::string& poolName);

  /* JIT Session */
private:
  static constexpr size_t LAUNCH_WINDOW_SIZE = 4;
  /// Number of ongoing kernel executions. This counter increments upon
  /// `enqueueKernelExecution` and decrements in the onFinish callback.
  std::atomic<unsigned> ongoings = 0;
  std::atomic<bool> timingEnabled = false;

  // Thread-safe lookup table. Each `LaunchTask` is created by
  // `enqueueKernelExecution` and owned by this `LaunchHistory`.
  struct LaunchHistory {
  private:
    struct Comparator {
      using is_transparent = void;

      bool operator()(const std::unique_ptr<LaunchTask>& a,
                      const std::unique_ptr<LaunchTask>& b) const noexcept {
        return a->id < b->id;
      }

      bool operator()(const std::unique_ptr<LaunchTask>& a,
                      const LaunchTask* b) const noexcept {
        return a->id < b->id;
      }

      bool operator()(const LaunchTask* a,
                      const std::unique_ptr<LaunchTask>& b) const noexcept {
        return a->id < b->id;
      }

      bool operator()(const std::unique_ptr<LaunchTask>& a,
                      size_t id) const noexcept {
        return a->id < id;
      }

      bool operator()(size_t id,
                      const std::unique_ptr<LaunchTask>& b) const noexcept {
        return id < b->id;
      }
    };

    std::set<std::unique_ptr<LaunchTask>, Comparator> data;
    std::mutex mtx;

  public:
    // Insert under the lock
    void insert(std::unique_ptr<LaunchTask> task) {
      std::lock_guard lk(mtx);
      data.insert(std::move(task));
    }

    // Lookup under the lock
    template <typename T> LaunchTask* lookup(const T& key) {
      std::lock_guard lk(mtx);
      auto it = data.find(key);
      if (it != data.end())
        return it->get();
      return nullptr;
    }
  } launchHistory_;

  LaunchTask* lookupLaunchTask(LaunchTask* ptr);

  struct LaunchSlot {
    LaunchTask* task = nullptr;

    CUmodule cuModule = nullptr;
    CUfunction cuFunction = nullptr;
    CUevent startEvent = nullptr;
    CUevent stopEvent = nullptr;

    CUdeviceptr argSvPtr = 0;
    CUdeviceptr argMatPtr = 0;
    size_t argCombosPtr = 0;
    std::array<void*, 3> args{};

    std::atomic<bool> finished = true;

    struct OnFinishUserData {
      std::atomic<bool>* finishedFlag = nullptr;
      std::atomic<unsigned>* ongoings = nullptr;
      CUevent* startEvent = nullptr;
      CUevent* stopEvent = nullptr;
      float* kernelTimeMs = nullptr;
    } onFinishUserData;

    LaunchSlot() {
      args[0] = &argSvPtr;
      args[1] = &argMatPtr;
      args[2] = &argCombosPtr;
      onFinishUserData.finishedFlag = &finished;
      onFinishUserData.startEvent = &startEvent;
      onFinishUserData.stopEvent = &stopEvent;
    }

    void resetResources() {
      if (cuModule != nullptr) {
        CU_CHECK(cuModuleUnload(cuModule));
        cuModule = nullptr;
      }
      if (startEvent != nullptr) {
        CU_CHECK(cuEventDestroy(startEvent));
        startEvent = nullptr;
      }
      if (stopEvent != nullptr) {
        CU_CHECK(cuEventDestroy(stopEvent));
        stopEvent = nullptr;
      }
      cuFunction = nullptr;
      task = nullptr;
    }

    void setArgs(CUdeviceptr sv, CUdeviceptr mat, size_t combos) {
      argSvPtr = sv;
      argMatPtr = mat;
      argCombosPtr = combos;
    }

    void** getArgs() { return args.data(); }

    void attachTask(LaunchTask* taskPtr, std::atomic<unsigned>* ongoingsPtr) {
      task = taskPtr;
      if (task != nullptr)
        task->kernelTimeMs = 0.0f;
      onFinishUserData.ongoings = ongoingsPtr;
      onFinishUserData.kernelTimeMs =
          task != nullptr ? &task->kernelTimeMs : nullptr;
    }

    static void CUDART_CB setFinishedCallback(void* ptr) {
      auto* userData = static_cast<OnFinishUserData*>(ptr);
      userData->finishedFlag->store(true, std::memory_order_release);
      userData->finishedFlag->notify_one();

      if (userData->kernelTimeMs != nullptr &&
          userData->startEvent != nullptr && userData->stopEvent != nullptr &&
          *userData->startEvent != nullptr && *userData->stopEvent != nullptr) {
        float ms = 0.0f;
        if (cuEventElapsedTime(&ms,
                               *userData->startEvent,
                               *userData->stopEvent) == CUDA_SUCCESS) {
          *userData->kernelTimeMs = ms;
        } else {
          *userData->kernelTimeMs = 0.0f;
        }
      }
      userData->ongoings->fetch_sub(1, std::memory_order_acq_rel);
      userData->ongoings->notify_all();
    }
  };

  /// A fixed-size window of launch slots for overlapping kernel launches.
  struct LaunchWindow {
  private:
    std::array<LaunchSlot, LAUNCH_WINDOW_SIZE> sessions{};

  public:
    LaunchWindow() = default;

    LaunchSlot& operator[](size_t idx) {
      return sessions[idx % LAUNCH_WINDOW_SIZE];
    }

    const LaunchSlot& operator[](size_t idx) const {
      return sessions[idx % LAUNCH_WINDOW_SIZE];
    }
  } launchWindow_;

  /// A queue with mutex and condition variable for launch tasks
  /// `LaunchQueue` does not own the `LaunchTask`s.
  struct LaunchQueue {
    // Need mtx + cv here because both the main thread and exec thread will
    // access this queue.
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<LaunchTask*> data;
  } launchQueue_;

  std::thread execTh_;
  std::atomic<bool> execThreadStopFlag = false;
  void execThreadFunc_();

  void startExecThread_() {
    execTh_ = std::thread(&CUDAKernelManager::execThreadFunc_, this);
  }

  void stopExecThread_() {
    execThreadStopFlag.store(true, std::memory_order_release);
    launchQueue_.cv.notify_one();
    if (execTh_.joinable())
      execTh_.join();
  }

public:
  /// Wait for all enqueued kernel compilations to finish.
  llvm::Error syncCompilation();

  void attachStatevector(std::unique_ptr<CUDAStatevectorBase> sv) {
    cuMgr.sv = std::move(sv);
  }

  void enableTiming(bool enable = true) {
    timingEnabled.store(enable, std::memory_order_release);
  }

  /// Enqueue a kernel for execution. Internally this function creates a
  /// `LaunchTask` that will then be moved into this manager's `launchHistory_`
  /// for memory management. During the entire launch and execution process, the
  /// launch task instance will only be referenced.
  llvm::Expected<LaunchTaskHandler> enqueueKernelExecution(CudaKernel* kernel);

  llvm::Error syncKernelExecution();

}; // class CUDAKernelManager

} // namespace cast

#endif // CAST_CUDA_CUDAKERNELMANAGER_H
