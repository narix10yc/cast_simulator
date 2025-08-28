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
// - cuContext: via CUDAKernelManager::primaryCuCtx (a single CUcontext)
// - cuModule: via CUDAKernelManager::cuModules (a vector of CUmodule)
// - cuFunction: living in its own CUmodule
struct CUDAKernelInfo {
  std::string ptxString;
  std::vector<uint8_t> cubinData;
  ConstQuantumGatePtr gate;
  Precision precision;
  llvm::LLVMContext* llvmContext;
  llvm::Module* llvmModule;
  std::string llvmFuncName;
  CUcontext cuContext;
  CUmodule cuModule;
  CUfunction cuFunction;

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
    JIT_PTXEmitted,
    JIT_CUBIN,
    JIT_CUFunctionLoaded
  };
  JITState jitState;
  int nWorkerThreads_;

  CUcontext primaryCuCtx;
  std::vector<CUmodule> cuModules;

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

public:
  CUDAKernelManager(int nWorkerThreads = -1)
      : KernelManagerBase(), standaloneKernels_(), jitState(JIT_Uninited),
        nWorkerThreads_(nWorkerThreads) {
    if (nWorkerThreads <= 0)
      this->nWorkerThreads_ = cast::get_cpu_num_threads();
  }

  CUDAKernelManager(const CUDAKernelManager&) = delete;
  CUDAKernelManager(CUDAKernelManager&&) = delete;
  CUDAKernelManager& operator=(const CUDAKernelManager&) = delete;
  CUDAKernelManager& operator=(CUDAKernelManager&&) = delete;

  ~CUDAKernelManager() {
    for (const auto& kernel : *this) {
      if (kernel.cuModule)
        cuModuleUnload(kernel.cuModule);
    }
    if (primaryCuCtx)
      cuCtxDestroy(primaryCuCtx);
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
  void compileLLVMIRToPTX(int llvmOptLevel = 1, int verbose = 0);

  void dumpPTX(std::ostream& os, const std::string& kernelName) const;

  // cuOptLevel: 0, 1, 2, 3, 4
  void compilePTXToCubin(int cuOptLevel = 1, int verbose = 0);

  void loadCubin(int verbose = 0);

  void clearPTX() {
    for (auto& kernel : *this)
      kernel.ptxString.clear();
  }

  void clearCubin() {
    for (auto& kernel : *this)
      kernel.cubinData.clear();
  }

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

  ///
  void launchCUDAKernel(void* dData,
                        int nQubits,
                        const CUDAKernelInfo& kernelInfo,
                        int blockDim = 64);

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