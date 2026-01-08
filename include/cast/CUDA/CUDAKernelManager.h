/*
 * CUDAKernelManager.h
 */
#ifndef CAST_CUDA_CUDAKERNELMANAGER_H
#define CAST_CUDA_CUDAKERNELMANAGER_H

#include "cast/CPU/Config.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "cast/Core/IRNode.h"
#include "cast/Core/KernelManager.h"
#include "cast/Core/QuantumGate.h"

#include "utils/InfoLogger.h"
#include "utils/ThreadPool.h"

#include <llvm/Support/Error.h>

#include <atomic>
#include <chrono>

#include <cuda.h>

namespace cast {

enum class CUDAMatrixLoadMode {
  UseMatImmValues,
  LoadInDefaultMemSpace,
  LoadInConstMemSpace
};

struct CUDAKernelInfo {
  std::string ptxString;
  std::vector<uint8_t> cubinData;
  ConstQuantumGatePtr gate = nullptr;
  Precision precision = Precision::Unknown;
  llvm::Function* llvmFunc = nullptr;

  void update(ConstQuantumGatePtr gate,
              Precision precision,
              llvm::Function* llvmFunc) {
    this->gate = gate;
    this->precision = precision;
    this->llvmFunc = llvmFunc;
  }

  void clearPTX() { ptxString.clear(); }
  void clearCubin() { cubinData.clear(); }

  std::string_view getName() const { return llvmFunc->getName(); }

  void displayInfo(utils::InfoLogger logger) const;
}; // struct CUDAKernelInfo

struct CUDAKernelGenConfig {
  Precision precision;
  double zeroTol = 1e-8;
  double oneTol = 1e-8;

  CUDAMatrixLoadMode matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;

  explicit CUDAKernelGenConfig(Precision p = Precision::FP64) : precision(p) {}

  void displayInfo(utils::InfoLogger logger) const;
};

class CUDAKernelManager : public KernelManager<CUDAKernelInfo> {
  using clock_t = std::chrono::steady_clock;
  using time_point_t = clock_t::time_point;

private:
  utils::ThreadPool<> tPool;

  CUdevice cuDevice;
  CUcontext primaryCuCtx = nullptr;
  CUstream primaryCuStream = nullptr;

  // The initializer
  void init();

  // Both gate-sv and superop-dm simulation will boil down to a matrix with
  // target qubits. This function contains the core logics to emit LLVM IR.
  llvm::Expected<llvm::Function*>
  gen_(const CUDAKernelGenConfig& config,
       const ComplexSquareMatrix& matrix,
       const QuantumGate::TargetQubitsType& qubits,
       const std::string& funcName,
       llvm::Module& llvmModule);

  /// An internal function to generate a CUDA kernel and put the kerne in the
  /// specified pool. This function wraps whether gate is a StandardQuantumGate
  /// (with or without noise) or SuperopQuantumGate, and call `gen_` with a
  /// corresponding ComplexSquareMatrix. The generated kernel will be put into
  /// the given pool.
  /// @param funcName: must be unique in the pool as should be guaranteed by the
  /// caller.
  llvm::Error genCUDAGate_(const CUDAKernelGenConfig& config,
                           ConstQuantumGatePtr gate,
                           const std::string& funcName,
                           Pool& pool);

  /* Kernel Execution Result */
public:
  struct ExecutionResult {
    std::string kernelName;

    // when the loading thread starts preparing the kernel
    time_point_t t_cubinPrepareStart;
    // when the loading thread finishes preparing the kernel
    time_point_t t_cubinPrepareFinish;

    // kernel execution time in milliseconds
    // Negative values indicate timing not enabled
    float kernelTime_ms = -1.0f;

    enum Status {
      // The initial launch status
      Pending,
      // Marked by loading threads after cubin is ready
      ReadyToLaunch,
      // Marked by the exec thread after cuLaunchKernel
      Running,
      // Marked by the cuda callback thread after kernel finishes
      Finished,
      // Marked by the exec thread after unloading the module
      CleanedUp
    };

    std::atomic<Status> status = Pending;

    ExecutionResult() = default;

    void displayInfo(utils::InfoLogger logger) const;

    // Get the compilation time in seconds
    float getCompileTime() const {
      std::chrono::duration<float> t(t_cubinPrepareFinish -
                                     t_cubinPrepareStart);
      return t.count();
    }

    // Get the kernel time in seconds. Returns 0.0f if timing is not enabled.
    float getKernelTime() const {
      if (kernelTime_ms <= 0.0f)
        return 0.0f;
      return kernelTime_ms * 1e-3f;
    }
  };

private:
  std::deque<ExecutionResult> execResults_;

  /* Launch Task Management */
private:
  struct InitialLaunch {
    CUDAKernelInfo* kernel;
    ExecutionResult* er;
  }; // struct InitialLaunch

  struct KernelSemaphore {
    std::mutex mtx{};
    std::condition_variable cv{};
    enum Status { Pending, Compiling, Prepared };
    Status status = Pending;
    KernelSemaphore() = default;
  };

  struct InitialLaunchQueue {
  private:
    std::mutex mtx_;
    std::deque<InitialLaunch> queue_;
    // There should be a reason we don't use shared_ptr here (can't remember)
    struct Semaphore {
      std::unique_ptr<KernelSemaphore> semaphore;
      int refCount;
    };
    std::map<CUDAKernelInfo*, Semaphore> semaphores_;

  public:
    KernelSemaphore* push(CUDAKernelInfo* kernel, ExecutionResult* er) {
      std::lock_guard lock(mtx_);
      queue_.emplace_back(kernel, er);
      if (semaphores_.find(kernel) == semaphores_.end())
        semaphores_[kernel] = {std::make_unique<KernelSemaphore>(), 1};
      else
        semaphores_[kernel].refCount += 1;
      return semaphores_[kernel].semaphore.get();
    }

    void pop(CUDAKernelInfo*& outKernel, ExecutionResult*& outEr) {
      std::lock_guard lock(mtx_);
      assert(!queue_.empty());
      outKernel = queue_.front().kernel;
      outEr = queue_.front().er;
      queue_.pop_front();
      auto spIt = semaphores_.find(outKernel);
      assert(spIt != semaphores_.end());
      spIt->second.refCount -= 1;
      if (spIt->second.refCount == 0)
        semaphores_.erase(spIt);
    }

    bool hasReadyToLaunch() {
      std::lock_guard lock(mtx_);
      if (queue_.empty())
        return false;
      return queue_.front().er->status.load() == ExecutionResult::ReadyToLaunch;
    }

    bool empty() {
      std::lock_guard lock(mtx_);
      return queue_.empty();
    }

    size_t size() {
      std::lock_guard lock(mtx_);
      return queue_.size();
    }

    void clear() {
      std::lock_guard lock(mtx_);
      queue_.clear();
    }
  };

  InitialLaunchQueue initialLaunches_;

  // execution thread
  std::thread execTh_;
  bool execStopFlag_ = false;
  std::mutex execMtx_;

  std::condition_variable execCV_;
  std::condition_variable syncCV_;

  enum SyncStatus { NotSyncing, RequestSyncing, Synced };
  // To load and store under execMtx_
  SyncStatus syncFlag_ = NotSyncing;

  // We need launch config because it is shared between the thread that poses
  // launch requests and the execution threads that actually launches the
  // kernels.
  struct LaunchConfig {
    CUdeviceptr devicePtr = 0;
    int nQubitsSV = 0;
    unsigned blockSize = 0;
  };

  LaunchConfig launchConfig_;

  // timing enabled by default
  bool timingEnabled_ = true;

  // execution thread work
  void execTh_work_();

public:
  CUDAKernelManager() : tPool(cast::get_cpu_num_threads()) { init(); }

  CUDAKernelManager(int nWorkerThreads) : tPool(nWorkerThreads) { init(); }

  CUDAKernelManager(const CUDAKernelManager&) = delete;
  CUDAKernelManager(CUDAKernelManager&&) = delete;
  CUDAKernelManager& operator=(const CUDAKernelManager&) = delete;
  CUDAKernelManager& operator=(CUDAKernelManager&&) = delete;

  ~CUDAKernelManager();

  void displayInfo(utils::InfoLogger logger) const;

  // Generate a kernel for a single gate into the default kernel pool.
  // \c funcName: if empty, a default name "k_<index>" will be assigned. If
  // provided, it must be unique among all kernels in the default pool.
  llvm::Error genGate(const CUDAKernelGenConfig& config,
                      ConstQuantumGatePtr gate,
                      const std::string& funcName = "");

  /// Generate kernels for all gates in the given circuit graph. The generated
  /// kernels will be named as <graphName>_<order>_<gateId>, where order is
  /// the order of the gate in the circuit graph.
  // TODO: do we still need the order
  llvm::Error genGraphGates(const CUDAKernelGenConfig& config,
                            const ir::CircuitGraphNode& graph,
                            const std::string& poolName);

  void dumpPTX(std::ostream& os, const std::string& kernelName) const;

  void clearPTX() {
    for (auto& kernel : all_kernels())
      kernel->clearPTX();
  }

  void clearCubin() {
    for (auto& kernel : all_kernels())
      kernel->clearCubin();
  }

  CUcontext getPrimaryCUContext() const { return primaryCuCtx; }
  CUstream getPrimaryCUStream() const { return primaryCuStream; }

  /* Get Kernels */

  unsigned numKernels() {
    unsigned count = 0;
    for (const auto& [name, items] : kernelPools_)
      count += items.size();
    return count;
  }

  void clearGraphKernels(const std::string& graphName) {
    auto it = kernelPools_.find(graphName);
    if (it != kernelPools_.end())
      kernelPools_.erase(it);
  }

  // Get kernel by name. Return nullptr if not found.
  CUDAKernelInfo* getKernelByName(const std::string& llvmFuncName);

  // Get kernel by name. Return nullptr if not found.
  const CUDAKernelInfo* getKernelByName(const std::string& llvmFuncName) const;

  /* --- Kernel Launch --- */

  // No lock/atomic needed because this is only called by the main thread.
  // launchConfig_ is only accessed after calling setLaunchConfig().
  void
  setLaunchConfig(CUdeviceptr dData, int nQubitsSV, unsigned blockSize = 64) {
    // Setting launch config with ongoing kernel launches is not allowed.
    // Potential inconsistency between when posing launch requests and actual
    // kernel execution
    assert(tPool.isIdle());

    // For compatibility (python api), we sync here
    tPool.sync();

    launchConfig_.devicePtr = dData;
    launchConfig_.nQubitsSV = nQubitsSV;
    launchConfig_.blockSize = blockSize;
  }

  bool isLaunchConfigValid() const {
    if (launchConfig_.devicePtr == 0)
      return false;
    if (launchConfig_.nQubitsSV <= 0)
      return false;

    switch (launchConfig_.blockSize) {
    case 32:
    case 64:
    case 128:
    case 256:
    case 512:
    case 1024:
      return true;
    default:
      return false;
    }
  }

  void setLaunchConfig(CUDAStatevectorFP32& sv, unsigned blockSize = 64) {
    setLaunchConfig(sv.getDevicePtr(), sv.nQubits(), blockSize);
  }

  void setLaunchConfig(CUDAStatevectorFP64& sv, unsigned blockSize = 64) {
    setLaunchConfig(sv.getDevicePtr(), sv.nQubits(), blockSize);
  }

  void enableTiming(bool enable = true) { timingEnabled_ = enable; }

  void clearExecutionResults() { execResults_.clear(); }

  /// Enqueue a kernel for launch. This is a non-blocking function and should
  /// only be called by the main thread. The order of kernel execution will
  /// align with the order of calling this function. Use `syncKernelExecution()`
  /// to wait for all kernels to finish running. The returned object is
  /// read-only, and should only be accessed after `syncKernelExecution()`. The
  /// returned object is invalidated upon the destruction of this kernel manager
  /// or upon calling `clearExecutionResults()`.
  /// @remark: We do not embed the ExecutionResult inside CUDAKernelInfo
  /// because users are allowed to launch the same kernel multiple times (for
  /// example, in benchmarking).
  const ExecutionResult* enqueueKernelLaunch(CUDAKernelInfo& kernel);

  float getTotalExecTime() const {
    float total = 0.0f;
    for (const auto& er : execResults_)
      total += er.getKernelTime();
    return total;
  }

  std::vector<const ExecutionResult*>
  enqueueKernelLaunchesFromGraph(const std::string& graphName) {
    auto it = kernelPools_.find(graphName);
    if (it == kernelPools_.end())
      return {}; // empty vector

    std::vector<const ExecutionResult*> results;
    results.reserve(it->second.size());

    for (auto& item : it->second) {
      const auto* res = enqueueKernelLaunch(*item.kernel);
      if (res != nullptr)
        results.push_back(res);
    }
    return results;
  }

  /// A blocking method that waits for all enqueued kernels to finish
  /// execution. This should only be called by the main thread.
  void syncKernelExecution(bool progressBar = false);

  // TODO: not implemented yet
  void launchCUDAKernelParam(void* dData,
                             int nQubits,
                             const CUDAKernelInfo& kernelInfo,
                             void* dMatPtr,
                             int blockDim = 64) {
    assert(false && "Not implemented yet");
  }
}; // class CUDAKernelManager

} // namespace cast

#endif // CAST_CUDA_CUDAKERNELMANAGER_H