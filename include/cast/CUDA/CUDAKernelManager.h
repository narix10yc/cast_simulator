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
  llvm::LLVMContext* llvmContext;
  llvm::Module* llvmModule;
  std::string llvmFuncName;

  std::ostream& displayInfo(std::ostream& os) const;
}; // struct CUDAKernelInfo

struct CUDAKernelGenConfig {
  Precision precision = Precision::F64; // default to double precision
  double zeroTol = 1e-8;
  double oneTol = 1e-8;

  CUDAMatrixLoadMode matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;

  std::ostream& displayInfo(std::ostream& os) const;
};

class CUDAKernelManager : public KernelManagerBase {
  using KernelInfoPtr = std::unique_ptr<CUDAKernelInfo>;

  std::vector<KernelInfoPtr> standaloneKernels_;
  std::map<std::string, std::vector<KernelInfoPtr>> graphKernels_;

  CUdevice cuDevice;
  CUcontext primaryCuCtx = nullptr;
  CUstream primaryCuStream = nullptr;

  // Both gate-sv and superop-dm simulation will boil down to a matrix with
  // target qubits. This function contains the core logics to emit LLVM IR.
  llvm::Function* gen_(const CUDAKernelGenConfig& config,
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

    llvm::Error err = llvm::Error::success();
    std::string kernelName;

    // when the loading thread starts preparing the kernel
    time_point_t t_cubinPrepareStart;
    // when the loading thread finishes preparing the kernel
    time_point_t t_cubinPrepareFinish;

    // kernel execution time in milliseconds
    float kernelTime_ms{};

    ExecutionResult() = default;

    std::ostream& displayInfo(std::ostream& os) const;

    // Get the compilation time in seconds
    float getCompileTime() const {
      return (t_cubinPrepareFinish - t_cubinPrepareStart).count() * 1e-3f;
    }

    // Get the kernel time in seconds (from CUevent)
    float getKernelTime() const { return kernelTime_ms * 1e-3f; }
  };

private:
  std::deque<ExecutionResult> execResults_;

public:
  /* Launch Task Management */
private:
  struct LaunchTask {
    std::vector<std::unique_ptr<std::byte[]>> kernelParamsStorage_;
    mutable std::vector<void*> kernelParams_;

    CUmodule cuModule = nullptr;
    CUfunction cuFunction = nullptr;

    CUevent startEvent = nullptr;
    CUevent finishEvent = nullptr;

    unsigned gridSize = 0;
    unsigned blockSize = 0;

    enum Status : int {
      // The task window is idle. This happens at the beginning of kernel
      // launch.
      Idle = 0,
      // The task window is being prepared by some loading thread. If some
      // loading thread find its allocated task window is in status Compiling,
      // it should wait till it turns to Running.
      Compiling = 1,
      // The task window is ready to be launched. A loading thread will set its
      // status to Ready, notify the execution thread, and fetch the next kernel
      // and task window.
      Ready = 2,
      // The task window is currently running. The execution thread launches a
      // Ready task and set its status to Running.
      Running = 3,
    };

    std::atomic<Status> status = Idle;

    // history will only be accessed by one loading thread at the same time.
    ExecutionResult* history = nullptr;

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
    unsigned windowSize;
    // This vector is initialized on the ctor of the kernel manager and should
    // never be resized during its lifetime.
    std::vector<LaunchTask> tasks_;

    LaunchTask& operator[](unsigned i) { return tasks_[i % windowSize]; }
    const LaunchTask& operator[](unsigned i) const {
      return tasks_[i % windowSize];
    }

    LaunchWindow(unsigned windowSize = 0)
        : windowSize(windowSize), tasks_(windowSize) {}
  }; // struct LaunchWindow

  LaunchWindow window_;

  // Clean up CUmodule and CUevents in the window. This function should only
  // be called after all kernels finishes execution. i.e. after
  // cuStreamSynchronize. It will also reset all LaunchTask status to
  // Uninited.
  void clearWindow_();

  // We do not need atomic here because launchIdx and loadIdx are only modified
  // in the exec thread.
  unsigned launchIdx = 0; // monotonically increasing index
  unsigned loadIdx = 0;   // monotonically increasing index

  /* Multi-threading task dispatching model:
  The main thread poses tasks via enqueueKernelLaunch. It enqueues a kernel
  loading task to the task dispatcher (loading thread pool).

  Each loading thread gets allocated with a launch window (given by loadIdx),
  loads the kernels, waits for the previous launch task to finish, unloads the
  previous task's module and emplaces the new task's module. Loading threads
  cannot directly launch kernels because we need to maintain the launch order.
  After doing all this, the loading thread sets the launch task status as
  'Ready' and notifies the execution thread that it has prepared the function
  and kernel parameters inside the launch window.

  The execution thread will launch ordered kernels. Every time it wakes up (via
  execCV_), it launches every task whose status is Ready. Kernel launch order is
  controlled by launchIdx.

  */

  // execution thread
  std::thread execTh_;
  bool stopFlag_ = false;
  std::mutex execMtx_;
  std::condition_variable execCV_;
  std::condition_variable loadCV_;
  std::condition_variable syncCV_;

  struct LaunchConfig {
    CUdeviceptr devicePtr = 0;
    int nQubits = 0;
    unsigned blockSize = 0;
  };

  LaunchConfig launchConfig_;

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

  llvm::Expected<const CUDAKernelInfo*>
  genStandaloneGate(const CUDAKernelGenConfig& config,
                    ConstQuantumGatePtr gate,
                    const std::string& funcName);

  /// Generate kernels for all gates in the given circuit graph. The generated
  /// kernels will be named as <graphName>_<order>_<gateId>, where order is the
  /// order of the gate in the circuit graph.
  // TODO: do we still need the order
  llvm::Error genGraphGates(const CUDAKernelGenConfig& config,
                            const ir::CircuitGraphNode& graph,
                            const std::string& graphName);

  // llvmOptLevel: 0, 1, 2, 3
  llvm::Error compileLLVMIRToPTX(int llvmOptLevel = 1, int verbose = 0);

  void dumpPTX(std::ostream& os, const std::string& kernelName) const;

  // cuOptLevel: 0, 1, 2, 3, 4
  llvm::Error compilePTXToCubin(int cuOptLevel = 1, int verbose = 0);

  llvm::Error initJIT(int optLevel = 1, int verbose = 0);

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
    for (const auto& [name, kernels] : graphKernels_)
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
    auto it = graphKernels_.find(graphName);
    if (it == graphKernels_.end())
      return {}; // empty span
    return std::span<const KernelInfoPtr>(it->second);
  }

  /* --- Kernel Launch --- */

  void
  setLaunchConfig(CUdeviceptr dData, int nQubits, unsigned blockSize = 64) {
    launchConfig_.devicePtr = dData;
    launchConfig_.nQubits = nQubits;
    launchConfig_.blockSize = blockSize;
  }

  /// Enqueue a kernel for launch. This is a non-blocking function and should
  /// only be called by the main thread. The order of kernel execution will
  /// align with the order of calling this function. Use \c
  /// syncKernelExecution() to wait for all kernels to finish running.
  /// The returned object is read-only, and should only be accessed after
  /// syncKernelExecution(). The returned object is invalidated upon the
  /// destruction of this kernel manager.
  /// @remark: We do not embed the ExecutionResult
  /// inside CUDAKernelInfo because users are allowed to launch the same kernel
  /// multiple times (for example, in benchmarking).
  const ExecutionResult* enqueueKernelLaunch(CUDAKernelInfo& kernel,
                                             int verbosity = 0);

  std::vector<const ExecutionResult*>
  enqueueKernelLaunchFromGraph(const std::string& graphName,
                               int verbosity = 0) {
    auto it = graphKernels_.find(graphName);
    if (it == graphKernels_.end())
      return {}; // empty vector

    std::vector<const ExecutionResult*> results;
    results.reserve(it->second.size());

    for (auto& kernel : it->second) {
      auto* res = enqueueKernelLaunch(*kernel, verbosity);
      if (res != nullptr)
        results.push_back(res);
    }
    return results;
  }

  /// A blocking method that waits for all enqueued kernels to finish execution.
  /// This should only be called by the main thread.
  void syncKernelExecution(bool progressBar = false);

  // TODO: not implemented yet
  void launchCUDAKernelParam(void* dData,
                             int nQubits,
                             const CUDAKernelInfo& kernelInfo,
                             void* dMatPtr,
                             int blockDim = 64);

  /* --- iterator ---*/
public:
  class iterator {
    using VecOfKernel = std::vector<KernelInfoPtr>;
    using MapOfKernels = std::map<std::string, VecOfKernel>;
    VecOfKernel::iterator vecIt, vecEnd;
    MapOfKernels::iterator mapIt, mapEnd;

    void proceedToNextMap() {
      while (mapIt != mapEnd) {
        vecIt = mapIt->second.begin();
        vecEnd = mapIt->second.end();
        if (vecIt != vecEnd) // found a non-empty one
          return;
        ++mapIt;
      }
      // reached the end
      vecIt = VecOfKernel::iterator();
      vecEnd = VecOfKernel::iterator();
    }

  public:
    using value_type = CUDAKernelInfo;
    iterator() = default;

    iterator(VecOfKernel::iterator vecIt,
             VecOfKernel::iterator vecEnd,
             MapOfKernels::iterator mapIt,
             MapOfKernels::iterator mapEnd)
        : vecIt(vecIt), vecEnd(vecEnd), mapIt(mapIt), mapEnd(mapEnd) {
      if (vecIt == vecEnd)
        proceedToNextMap();
    }

    value_type& operator*() const { return *(vecIt->get()); }
    value_type* operator->() const { return &**this; }

    iterator& operator++() {
      ++vecIt;
      if (vecIt == vecEnd)
        proceedToNextMap();

      return *this;
    }

    iterator operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    bool operator==(const iterator& other) const {
      return vecIt == other.vecIt && mapIt == other.mapIt;
    }

    bool operator!=(const iterator& other) const { return !(*this == other); }
  }; // iterator

  iterator begin() {
    return iterator(standaloneKernels_.begin(),
                    standaloneKernels_.end(),
                    graphKernels_.begin(),
                    graphKernels_.end());
  }

  iterator end() {
    return iterator(standaloneKernels_.end(),
                    standaloneKernels_.end(),
                    graphKernels_.end(),
                    graphKernels_.end());
  }

  /* --- const iterator ---*/
public:
  class const_iterator {
    using VecOfKernel = std::vector<KernelInfoPtr>;
    using MapOfKernels = std::map<std::string, VecOfKernel>;
    VecOfKernel::const_iterator vecIt, vecEnd;
    MapOfKernels::const_iterator mapIt, mapEnd;

    void proceedToNextMap() {
      while (mapIt != mapEnd) {
        vecIt = mapIt->second.begin();
        vecEnd = mapIt->second.end();
        if (vecIt != vecEnd) // found a non-empty one
          return;
        ++mapIt;
      }
      // reached the end
      vecIt = VecOfKernel::const_iterator();
      vecEnd = VecOfKernel::const_iterator();
    }

  public:
    using value_type = const CUDAKernelInfo;
    const_iterator() = default;

    const_iterator(VecOfKernel::const_iterator vecIt,
                   VecOfKernel::const_iterator vecEnd,
                   MapOfKernels::const_iterator mapIt,
                   MapOfKernels::const_iterator mapEnd)
        : vecIt(vecIt), vecEnd(vecEnd), mapIt(mapIt), mapEnd(mapEnd) {
      if (vecIt == vecEnd)
        proceedToNextMap();
    }

    value_type& operator*() const { return *(vecIt->get()); }
    value_type* operator->() const { return &**this; }

    const_iterator& operator++() {
      ++vecIt;
      if (vecIt == vecEnd)
        proceedToNextMap();

      return *this;
    }

    const_iterator operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    bool operator==(const const_iterator& other) const {
      return vecIt == other.vecIt && mapIt == other.mapIt;
    }

    bool operator!=(const const_iterator& other) const {
      return !(*this == other);
    }
  }; // const _iterator

  const_iterator begin() const {
    return const_iterator(standaloneKernels_.begin(),
                          standaloneKernels_.end(),
                          graphKernels_.begin(),
                          graphKernels_.end());
  }

  const_iterator end() const {
    return const_iterator(standaloneKernels_.end(),
                          standaloneKernels_.end(),
                          graphKernels_.end(),
                          graphKernels_.end());
  }
};

} // namespace cast

#endif // CAST_CUDA_CUDAKERNELMANAGER_H