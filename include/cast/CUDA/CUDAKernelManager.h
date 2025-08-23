/*
 * CUDAKernelManager.h
 */
#ifndef CAST_CUDA_CUDAKERNELMANAGER_H
#define CAST_CUDA_CUDAKERNELMANAGER_H

#include "cast/CUDA/Config.h"
#include "cast/Core/IRNode.h"
#include "cast/Core/KernelManager.h"
#include "cast/Core/QuantumGate.h"

#include "utils/MaybeError.h"

#include <span>

// TODO: We may not need cuda_runtime (also no need to link CUDA::cudart)
#include <cuda.h>
#include <cuda_runtime.h>
#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <algorithm>


namespace cast {

enum class CUDAMatrixLoadMode {
  UseMatImmValues,
  LoadInDefaultMemSpace,
  LoadInConstMemSpace
};

struct BitLayout {
  std::vector<int> log_of_phys; // physical pos -> logical id
  std::vector<int> phys_of_log; // logical id  -> physical pos
  void init(int nSys) {
    log_of_phys.resize(nSys);
    phys_of_log.resize(nSys);
    std::iota(log_of_phys.begin(), log_of_phys.end(), 0);
    std::iota(phys_of_log.begin(), phys_of_log.end(), 0);
  }
  // After permuting gate qubits Q into {0..k-1} in-order, update maps.
  void setLSB(const std::vector<int>& Q, int nSys) {
    const int k = (int)Q.size();
    // non-target logicals, in ascending old physical order
    std::vector<std::pair<int,int>> others; others.reserve(nSys-k);
    std::vector<char> isTarget(nSys, 0);
    for (int b = 0; b < k; ++b) isTarget[Q[b]] = 1;
    for (int l = 0; l < nSys; ++l) if (!isTarget[l])
      others.emplace_back(phys_of_log[l], l);
    std::sort(others.begin(), others.end()); // by old phys pos

    // write new phys_of_log
    for (int b = 0; b < k; ++b)        phys_of_log[Q[b]] = b;
    for (int i = 0; i < (int)others.size(); ++i)
      phys_of_log[others[i].second] = k + i;

    // rebuild inverse
    for (int p = 0; p < nSys; ++p) log_of_phys[p] = -1;
    for (int l = 0; l < nSys; ++l) log_of_phys[phys_of_log[l]] = l;
  }
};

struct KernelKey {
  unsigned k;
  Precision prec;
  CUDAMatrixLoadMode load;
  bool assumeContiguous;
  uint64_t matHash;  // 0 for runtime-loaded matrices; hash for immediate path
  bool operator==(const KernelKey&) const = default;
};
struct KernelKeyHash {
  size_t operator()(const KernelKey& k) const {
    // cheap hash; feel free to xxhash
    size_t h = 1469598103934665603ull;
    auto mix=[&](uint64_t v){ h ^= v; h *= 1099511628211ull; };
    mix(k.k); mix((uint64_t)k.prec); mix((uint64_t)k.load);
    mix(k.assumeContiguous); mix(k.matHash);
    return h;
  }
};
struct KernelInfoCompiled {
  llvm::Function* fn = nullptr;
  std::string llvmFuncName;  // for launch lookup
};

struct CUDAKernelInfo {
  /// If CAST_USE_CUDA is not defined, \c CUDATuple is simply an empty struct.
  /// Every CUDA module will contain exactly one CUDA function. Multiple CUDA
  /// modules may share the same CUDA context. For multi-threading JIT session,
  /// we create \c nThread number of CUDA contexts.
  /// CUcontext internally is just a pointer. So multiple \c CUDATuple may have
  /// the same \c cuContext. The collection of unique \c cuContext is stored in
  /// \c cuContexts (only available if CAST_USE_CUDA is defined).
  struct CUDATuple {
#ifdef CAST_USE_CUDA
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;
    size_t sharedMemBytes;
    CUDATuple()
        : cuContext(nullptr), cuModule(nullptr), cuFunction(nullptr),
          sharedMemBytes(0) {}
#endif // #ifdef CAST_USE_CUDA
  };
  // We expect large stream writes anyway, so always trigger heap allocation.
  using PTXStringType = llvm::SmallString<0>;
  PTXStringType ptxString;
  Precision precision;
  std::string llvmFuncName;
  ConstQuantumGatePtr gate;
  CUDATuple cuTuple;
  int opCount;
  unsigned regsPerThread = 32;
  bool oneThreadPerBlock = false;
  unsigned tileSize = 0;
  unsigned warpsPerCTA = 0;
  std::string kstyle;
  // int blockSize = 256;

#ifdef CAST_USE_CUDA
  void setSharedMemUsage(size_t bytes) { cuTuple.sharedMemBytes = bytes; }
  void setKernelFunction(CUfunction fn) { cuTuple.cuFunction = fn; }
  CUfunction kernelFunction() const { return cuTuple.cuFunction; }
  size_t sharedMemUsage() const { return cuTuple.sharedMemBytes; }
  void setRegisterUsage(unsigned r) { regsPerThread = r; }
  unsigned registerUsage() const { return regsPerThread; }
#endif // #ifdef CAST_USE_CUDA
}; // struct CUDAKernelInfo

struct CUDAKernelGenConfig {
  Precision precision = Precision::F64; // default to double precision
  double zeroTol = 1e-8;
  double oneTol = 1e-8;
  bool forceDenseKernel = false;
  int blockSize = 64; // for now have constant blocksize across kernels
  bool assumeContiguousTargets = false;
  bool useAsyncTiles = true;

  // Enable tiling if the gate size >= this value. Setting this value to 0
  // always enables tiling.
  int enableTilingGateSize = 0;
  CUDAMatrixLoadMode matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;

  std::ostream& displayInfo(std::ostream& os) const;
};

class CUDAKernelManager : public KernelManagerBase {
  using KernelInfoPtr = std::unique_ptr<CUDAKernelInfo>;
  struct KernelPair {
    KernelInfoPtr lsb;  // assumeContiguousTargets = true
    KernelInfoPtr gen;  // assumeContiguousTargets = false
  };
  std::vector<KernelInfoPtr> standaloneKernels_;
  std::map<std::string, std::vector<KernelInfoPtr>> graphKernels_;

  enum JITState { JIT_Uninited, JIT_PTXEmitted, JIT_CUFunctionLoaded };
  JITState jitState;

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
  MaybeError<KernelPair> genCUDAGateVariants_(const CUDAKernelGenConfig& config,
                                              ConstQuantumGatePtr gate,
                                              const std::string& baseName);

public:
  CUDAKernelManager()
      : KernelManagerBase(), standaloneKernels_(), jitState(JIT_Uninited) {}

  std::unordered_map<KernelKey, KernelInfoCompiled, KernelKeyHash> kernelCache_;
  MaybeError<void> genStandaloneGate(const CUDAKernelGenConfig& config,
                                     ConstQuantumGatePtr gate,
                                     const std::string& funcName);

  void emitPTX(int nThreads = 1,
               llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O0,
               int verbose = 0);

  std::string getPTXString(int idx) const {
    return std::string(standaloneKernels_[idx]->ptxString.str());
  }

  void dumpPTX(std::ostream& os, const std::string& kernelName) const;

  /// Generate kernels for all gates in the given circuit graph. The generated
  /// kernels will be named as <graphName>_<order>_<gateId>, where order is the
  /// order of the gate in the circuit graph.
  // TODO: do we still need the order
  MaybeError<void> genGraphGates(const CUDAKernelGenConfig& config,
                                 const ir::CircuitGraphNode& graph,
                                 const std::string& graphName);

private:
  /// \c cuContexts stores a vector of unique \c CUContext. Every thread will
  /// manage its own CUcontext. These \c CUContext 's will be destructed upon
  /// the destruction of this class.
  std::vector<CUcontext> cuContexts;

public:
  CUDAKernelManager(const CUDAKernelManager&) = delete;
  CUDAKernelManager(CUDAKernelManager&&) = delete;
  CUDAKernelManager& operator=(const CUDAKernelManager&) = delete;
  CUDAKernelManager& operator=(CUDAKernelManager&&) = delete;

  ~CUDAKernelManager() {
  #ifdef CAST_USE_CUDA
    auto unload = [&](CUDAKernelInfo* k) {
      if (k && k->cuTuple.cuModule) {
        cuModuleUnload(k->cuTuple.cuModule);
      }
    };
    if (!orderedKernels_.empty()) {
      for (auto* k : orderedKernels_) unload(k);
    } else {
      for (auto& k : standaloneKernels_) unload(k.get());
      for (auto& kv : graphKernels_) {
        for (auto& k : kv.second) unload(k.get());
      }
    }
    for (auto& ctx : cuContexts) {
      if (ctx) cuCtxDestroy(ctx);
    }
  #else
    for (auto& ctx : cuContexts) {
      if (ctx) cuCtxDestroy(ctx);
    }
  #endif
  }

  /// @brief Initialize CUDA JIT session by loading PTX strings into CUDA
  /// context and module. This function can only be called once and cannot be
  /// undone. This function assumes \c emitPTX is already called.
  void initCUJIT(int nThreads = 1, int verbose = 0);


  /* Get Kernels */

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
                        int blockSize = 64);

  void launchCUDAKernelParam(void* dData,
                             int nQubits,
                             const CUDAKernelInfo& kernelInfo,
                             void* dMatPtr,
                             int blockSize = 64);
  std::vector<CUDAKernelInfo*> orderedKernels_;
  void rebuildOrderedKernelIndex_();
  KernelInfoCompiled getOrBuildKernel_(
    const CUDAKernelGenConfig& baseCfg,
    const ComplexSquareMatrix& M,
    llvm::ArrayRef<int> qubits,
    bool assumeContiguous,
    const std::string& nameHint);
};

} // namespace cast

#endif // CAST_CUDA_CUDAKERNELMANAGER_H