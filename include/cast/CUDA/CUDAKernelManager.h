/*
 * CUDAKernelManager.h
 */
#ifndef CAST_CUDA_CUDAKERNELMANAGER_H
#define CAST_CUDA_CUDAKERNELMANAGER_H

#include "cast/CUDA/Config.h"
#include "cast/Core/IRNode.h"
#include "cast/Core/KernelManager.h"
#include "cast/Core/QuantumGate.h"

#include "cast/CPU/Config.h" // for cast::get_cpu_num_threads()

#include "llvm/Support/Error.h"

#include <cuda.h>
#include <map>
#include <span>

namespace cast {

enum class CUDAMatrixLoadMode {
  UseMatImmValues,
  LoadInDefaultMemSpace,
  LoadInConstMemSpace
};

struct CUDAKernelInfo {

  std::string ptxString;
  std::vector<uint8_t> cubinData;
  ConstQuantumGatePtr gate;
  Precision precision;
  llvm::Function* llvmFunc;

  std::string_view getName() const { return llvmFunc->getName(); }

  std::ostream& displayInfo(std::ostream& os) const;
}; // struct CUDAKernelInfo

struct CUDAKernelGenConfig {
  Precision precision;
  double zeroTol = 1e-8;
  double oneTol = 1e-8;

  CUDAMatrixLoadMode matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;

  explicit CUDAKernelGenConfig(Precision p = Precision::FP64) : precision(p) {}

  std::ostream& displayInfo(std::ostream& os) const;
};

class CUDAKernelManager : public KernelManager<CUDAKernelInfo> {

  CUdevice cuDevice;
  CUcontext primaryCuCtx = nullptr;
  CUstream primaryCuStream = nullptr;

  // Both gate-sv and superop-dm simulation will boil down to a matrix with
  // target qubits. This function contains the core logics to emit LLVM IR.
  llvm::Expected<llvm::Function*>
  gen_(const CUDAKernelGenConfig& config,
       const ComplexSquareMatrix& matrix,
       const QuantumGate::TargetQubitsType& qubits,
       const std::string& funcName);

  // Generate a CUDA kernel for the given gate. This function will wraps around
  // when gate is a StandardQuantumGate (with or without noise) or
  // SuperopQuantumGate, and call gen_ with a corresponding ComplexSquareMatrix.
  llvm::Expected<KernelInfoPtr> genCUDAGate_(const CUDAKernelGenConfig& config,
                                             ConstQuantumGatePtr gate,
                                             const std::string& funcName);

  /* Kernel Execution Result */
public:
  struct ExecutionResult {
    using clock_t = std::chrono::steady_clock;
    using time_point_t = clock_t::time_point;

    // llvm::Error err = llvm::Error::success();
    std::string kernelName;

    // when the loading thread starts preparing the kernel
    time_point_t t_cubinPrepareStart;
    // when the loading thread finishes preparing the kernel
    time_point_t t_cubinPrepareFinish;

    // kernel execution time in milliseconds
    float kernelTime_ms{};

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

    std::ostream& displayInfo(std::ostream& os) const;

    // Get the compilation time in seconds
    float getCompileTime() const {
      std::chrono::duration<float> t(t_cubinPrepareFinish -
                                     t_cubinPrepareStart);
      return t.count();
    }

    // Get the kernel time in seconds (from CUevent)
    float getKernelTime() const { return kernelTime_ms * 1e-3f; }
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

  struct LaunchConfig {
    CUdeviceptr devicePtr = 0;
    int nQubitsSV = 0;
    unsigned blockSize = 0;
  };

  LaunchConfig launchConfig_;

  bool timingEnabled_ = false;

  // execution thread work
  void execTh_work_();

public:
  CUDAKernelManager(int nWorkerThreads = -1, int deviceOrdinal = 0);

  CUDAKernelManager(const CUDAKernelManager&) = delete;
  CUDAKernelManager(CUDAKernelManager&&) = delete;
  CUDAKernelManager& operator=(const CUDAKernelManager&) = delete;
  CUDAKernelManager& operator=(CUDAKernelManager&&) = delete;

  ~CUDAKernelManager();

  std::ostream& displayInfo(std::ostream& os) const;

  llvm::Expected<CUDAKernelInfo*>
  genStandaloneGate(const CUDAKernelGenConfig& config,
                    ConstQuantumGatePtr gate,
                    const std::string& funcName);

  /// Generate kernels for all gates in the given circuit graph. The generated
  /// kernels will be named as <graphName>_<order>_<gateId>, where order is
  /// the order of the gate in the circuit graph.
  // TODO: do we still need the order
  llvm::Error genGraphGates(const CUDAKernelGenConfig& config,
                            const ir::CircuitGraphNode& graph,
                            const std::string& graphName);

  void dumpPTX(std::ostream& os, const std::string& kernelName) const;

  void clearPTX() {
    for (auto& kernel : *this)
      kernel.ptxString.clear();
  }

  void clearCubin() {
    for (auto& kernel : *this)
      kernel.cubinData.clear();
  }

  CUcontext getPrimaryCUContext() const { return primaryCuCtx; }
  CUstream getPrimaryCUStream() const { return primaryCuStream; }

  /* Get Kernels */

  unsigned numKernels() {
    unsigned count = standaloneKernels_.size();
    for (const auto& [name, kernels] : kernelPools_)
      count += kernels.size();
    return count;
  }

  std::span<const KernelInfoPtr> getAllStandaloneKernels() const {
    return std::span<const KernelInfoPtr>(standaloneKernels_);
  }

  // Get kernel by name. Return nullptr if not found.
  CUDAKernelInfo* getKernelByName(const std::string& llvmFuncName);

  // Get kernel by name. Return nullptr if not found.
  const CUDAKernelInfo* getKernelByName(const std::string& llvmFuncName) const;

  std::span<const KernelInfoPtr>
  getKernelsFromGraphName(const std::string& graphName) const {
    auto it = kernelPools_.find(graphName);
    if (it == kernelPools_.end())
      return {}; // empty span
    return std::span<const KernelInfoPtr>(it->second);
  }

  /* --- Kernel Launch --- */

  void
  setLaunchConfig(CUdeviceptr dData, int nQubitsSV, unsigned blockSize = 64) {
    launchConfig_.devicePtr = dData;
    launchConfig_.nQubitsSV = nQubitsSV;
    launchConfig_.blockSize = blockSize;
  }

  void enableTiming(bool enable = true) { timingEnabled_ = enable; }

  void clearExecutionResults() { execResults_.clear(); }

  /// Enqueue a kernel for launch. This is a non-blocking function and should
  /// only be called by the main thread. The order of kernel execution will
  /// align with the order of calling this function. Use \c
  /// syncKernelExecution() to wait for all kernels to finish running.
  /// The returned object is read-only, and should only be accessed after
  /// syncKernelExecution(). The returned object is invalidated upon the
  /// destruction of this kernel manager.
  /// @remark: We do not embed the ExecutionResult inside CUDAKernelInfo because
  /// users are allowed to launch the same kernel multiple times (for example,
  /// in benchmarking).
  const ExecutionResult* enqueueKernelLaunch(CUDAKernelInfo& kernel);

  float getTotalExecTime() const {
    float total = 0.0f;
    for (const auto& er : execResults_)
      total += er.getKernelTime();
    return total;
  }

  std::vector<const ExecutionResult*>
  enqueueKernelLaunchFromGraph(const std::string& graphName) {
    auto it = kernelPools_.find(graphName);
    if (it == kernelPools_.end())
      return {}; // empty vector

    std::vector<const ExecutionResult*> results;
    results.reserve(it->second.size());

    for (auto& kernel : it->second) {
      const auto* res = enqueueKernelLaunch(*kernel);
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
                             int blockDim = 64);
}; // class CUDAKernelManager

} // namespace cast

#endif // CAST_CUDA_CUDAKERNELMANAGER_H