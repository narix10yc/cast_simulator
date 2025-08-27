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

struct CUDAKernelInfo {
  /// Every CUDA module will contain exactly one CUDA function. Multiple CUDA
  /// modules may share the same CUDA context. For multi-threading JIT session,
  /// we create \c nThread number of CUDA contexts.
  /// CUcontext internally is just a pointer. So multiple \c CUDATuple may have
  /// the same \c cuContext. The collection of unique \c cuContext is stored in
  /// \c cuContexts (only available if CAST_USE_CUDA is defined).
  struct CUDATuple {
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;
    CUDATuple()
        : cuContext(nullptr), cuModule(nullptr), cuFunction(nullptr) {}
  };
  // We expect large stream writes anyway, so always trigger heap allocation.
  using PTXStringType = llvm::SmallString<0>;
  PTXStringType ptxString;
  Precision precision;
  llvm::LLVMContext* llvmContext;
  llvm::Module* llvmModule;
  std::string llvmFuncName;
  ConstQuantumGatePtr gate;
  CUDATuple cuTuple;
  double opCount;

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

  enum JITState { JIT_Uninited, JIT_PTXEmitted, JIT_CUFunctionLoaded };
  JITState jitState;
  int nWorkerThreads_;

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

  MaybeError<void> genStandaloneGate(const CUDAKernelGenConfig& config,
                                     ConstQuantumGatePtr gate,
                                     const std::string& funcName);

  void emitPTX(llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O0,
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
    for (auto& [name, kernels] : graphKernels_) {
      for (auto& kernel : kernels) {
        if (kernel->cuTuple.cuModule) {
          cuModuleUnload(kernel->cuTuple.cuModule);
        }
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
  void initCUJIT(int verbose = 0);

  // cuOptLevel: 0 .. 4
  void initCUJIT_New(int cuOptLevel = 1, int verbose = 0);

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
                        int blockDim = 64);

  void launchCUDAKernelParam(void* dData,
                             int nQubits,
                             const CUDAKernelInfo& kernelInfo,
                             void* dMatPtr,
                             int blockDim = 64);
};

} // namespace cast

#endif // CAST_CUDA_CUDAKERNELMANAGER_H