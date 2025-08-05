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


namespace cast {

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

enum class CUDAMatrixLoadMode {
  UseMatImmValues,       // use immediate values
  LoadInDefaultMemSpace, // load in default memory space
  LoadInConstMemSpace    // load in constant memory space
};

struct CUDAKernelGenConfig {
  Precision precision = Precision::F64; // default to double precision
  double zeroTol = 1e-8;
  double oneTol = 1e-8;
  bool forceDenseKernel = false;
  int blockSize = 64; // for now have constant blocksize across kernels

  // Enable tiling if the gate size >= this value. Setting this value to 0
  // always enables tiling.
  int enableTilingGateSize = 0;
  CUDAMatrixLoadMode matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;

  std::ostream& displayInfo(std::ostream& os) const;
};

class CUDAKernelManager : public KernelManagerBase {
  using KernelInfoPtr = std::unique_ptr<CUDAKernelInfo>;
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

public:
  CUDAKernelManager()
      : KernelManagerBase(), standaloneKernels_(), jitState(JIT_Uninited) {}

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
    for (auto& kernel : standaloneKernels_) {
      if (kernel->cuTuple.cuModule) {
        cuModuleUnload(kernel->cuTuple.cuModule);
      }
    }
    for (auto& ctx : cuContexts) {
      if (ctx) {
        cuCtxDestroy(ctx);
      }
    }
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
};

} // namespace cast

#endif // CAST_CUDA_CUDAKERNELMANAGER_H