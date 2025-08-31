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
#include "utils/MaybeError.h"

#include <cuda.h>
#include <map>
#include <span>

namespace cast {

enum class CUDAMatrixLoadMode {
  UseMatImmValues,
  LoadInDefaultMemSpace,
  LoadInConstMemSpace
};

// cuContext, cuModule, and cuFunction are internally pointers. CUDAKernelInfo
// does not own their allocation state. The ownerships of these objects are
// - cuContext: via CUDAKernelManager::primaryCuCtx (a single CUcontext shared
//              across kernels)
// - cuModule: owned by CUDAKernelInfo::cuModule (one CUmodule per kernel)
// - cuFunction: living in its own CUmodule
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

  enum JITState {
    JIT_Uninited,
    JIT_CompiledPTX,
    JIT_CompiledCubin,
    JIT_CubinLoaded
  };
  JITState jitState = JIT_Uninited;

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
  MaybeError<KernelInfoPtr> genCUDAGate_(const CUDAKernelGenConfig& config,
                                         ConstQuantumGatePtr gate,
                                         const std::string& funcName);

private:
  struct LaunchTask {
  private:
    std::vector<std::unique_ptr<std::byte[]>> kernelParamsStorage_;
    mutable std::vector<void*> kernelParams_;

  public:
    CUmodule cuModule = nullptr;
    // cuFunction is used to check if this kernel is launch-able
    // After the executer thread launches this kernel, cuModule and cuFunction
    // should be set to null
    CUfunction cuFunction = nullptr;

    CUevent startEvent = nullptr;
    CUevent finishEvent = nullptr;

    unsigned gridSize = 0;
    unsigned blockSize = 0;

    enum Status : int {
      Uninited = 0, // The task is not initialized
      Ready = 1,    // The task is ready to be launched
      Running = 2,  // The task is currently running
      Finished = 3  // The task has finished
    };

    std::atomic<Status> status = Uninited;

    LaunchTask() = default;

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
  };
  struct Window {
    static constexpr int WS = 1; // window size
    std::array<LaunchTask, WS> tasks_;

    LaunchTask& operator[](unsigned i) { return tasks_[i % WS]; }
    const LaunchTask& operator[](unsigned i) const { return tasks_[i % WS]; }
  };

  std::deque<CUDAKernelInfo*> pending_; // kernels waiting to be loaded
  Window window_;
  // We do not need atomic here because launchIdx and loadIdx are only modified
  // in the exec thread.
  unsigned launchIdx = 0; // monotonically increasing index
  unsigned loadIdx = 0;   // monotonically increasing index
  std::thread execTh_;
  bool stopFlag_ = false;
  std::mutex execMtx_;
  std::condition_variable execCV_;

  std::condition_variable loadCV_;

  struct LaunchConfig {
    CUdeviceptr dData = 0;
    int nQubits = 0;
    unsigned blockSize = 0;
  };

  LaunchConfig launchConfig_;

  void execThreadWork();

public:
  CUDAKernelManager(int nWorkerThreads = -1, int deviceOrdinal = 0)
      : KernelManagerBase((nWorkerThreads > 0) ? nWorkerThreads
                                               : cast::get_cpu_num_threads()) {
    CU_CHECK(cuInit(0));
    CU_CHECK(cuDeviceGet(&cuDevice, deviceOrdinal));
    CU_CHECK(cuDevicePrimaryCtxRetain(&primaryCuCtx, cuDevice));
    CU_CHECK(cuCtxSetCurrent(primaryCuCtx));
    CU_CHECK(cuStreamCreate(&primaryCuStream, CU_STREAM_DEFAULT));

    // initialize exec thread
    execTh_ = std::thread(&CUDAKernelManager::execThreadWork, this);
  }

  CUDAKernelManager(const CUDAKernelManager&) = delete;
  CUDAKernelManager(CUDAKernelManager&&) = delete;
  CUDAKernelManager& operator=(const CUDAKernelManager&) = delete;
  CUDAKernelManager& operator=(CUDAKernelManager&&) = delete;

  ~CUDAKernelManager() {
    dispatcher.sync();
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

  std::ostream& displayInfo(std::ostream& os) const;

  MaybeError<void> genStandaloneGate(const CUDAKernelGenConfig& config,
                                     ConstQuantumGatePtr gate,
                                     const std::string& funcName);

  /// Generate kernels for all gates in the given circuit graph. The generated
  /// kernels will be named as <graphName>_<order>_<gateId>, where order is the
  /// order of the gate in the circuit graph.
  // TODO: do we still need the order
  MaybeError<void> genGraphGates(const CUDAKernelGenConfig& config,
                                 const ir::CircuitGraphNode& graph,
                                 const std::string& graphName);

  // llvmOptLevel: 0, 1, 2, 3
  MaybeError<void> compileLLVMIRToPTX(int llvmOptLevel = 1, int verbose = 0);

  void dumpPTX(std::ostream& os, const std::string& kernelName) const;

  // cuOptLevel: 0, 1, 2, 3, 4
  MaybeError<void> compilePTXToCubin(int cuOptLevel = 1, int verbose = 0);

  MaybeError<void> initJIT(int optLevel = 1, int verbose = 0);

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
    launchConfig_.dData = dData;
    launchConfig_.nQubits = nQubits;
    launchConfig_.blockSize = blockSize;
  }

  ///
  void enqueueKernelLaunch(CUDAKernelInfo& kernel) {
    {
      std::unique_lock lk(execMtx_);
      pending_.push_back(&kernel);
    }
    execCV_.notify_one();
  }

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