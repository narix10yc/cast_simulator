#ifndef CAST_CORE_KERNEL_MANAGER_H
#define CAST_CORE_KERNEL_MANAGER_H

#include "cast/Legacy/QuantumGate.h"
#include "cast/Core/Precision.h"

#include "llvm/IR/Module.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Passes/OptimizationLevel.h"

#include <memory>

#ifdef CAST_USE_CUDA
  #include <cuda.h>
  // TODO: We may not need cuda_runtime (also no need to link CUDA::cudart)
  #include <cuda_runtime.h>
#endif // CAST_USE_CUDA

namespace cast {

  namespace internal {
    /// mangled name is formed by 'G' + <length of graphName> + graphName
    /// @return mangled name
    std::string mangleGraphName(const std::string& graphName);

    std::string demangleGraphName(const std::string& mangledName);
  } // namespace internal

  // forward declaration
  namespace legacy {
    class CircuitGraph;
  }
  class CircuitGraph;

class KernelManagerBase {
protected:
  struct ContextModulePair {
    std::unique_ptr<llvm::LLVMContext> llvmContext;
    std::unique_ptr<llvm::Module> llvmModule;
  };
  // A vector of pairs of LLVM context and module. Expected to be cleared after
  // calling \c initJIT
  std::vector<ContextModulePair> llvmContextModulePairs;
  std::mutex mtx;

  /// A thread-safe version that creates a new llvm Module
  ContextModulePair& createNewLLVMContextModulePair(const std::string& name);

  /// Apply LLVM optimization to all modules inside \c llvmContextModulePairs
  /// As a private member function, this function will be called by \c initJIT
  /// and \c initJITForPTXEmission
  void applyLLVMOptimization(
      int nThreads, llvm::OptimizationLevel optLevel, bool progressBar);
};

/*
  CUDA
*/
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
      : cuContext(nullptr)
      , cuModule(nullptr)
      , cuFunction(nullptr)
      , sharedMemBytes(0)
    {}
    #endif // #ifdef CAST_USE_CUDA
  };
  // We expect large stream writes anyway, so always trigger heap allocation.
  using PTXStringType = llvm::SmallString<0>;
  PTXStringType ptxString;
  Precision precision;
  std::string llvmFuncName;
  std::shared_ptr<legacy::QuantumGate> gate;
  CUDATuple cuTuple;
  int opCount;
  unsigned regsPerThread = 32;
  // int blockSize = 256;
  #ifdef CAST_USE_CUDA
  void setGate(const legacy::QuantumGate* g) {
      gate = std::shared_ptr<legacy::QuantumGate>(
                 const_cast<legacy::QuantumGate*>(g),
                 [](legacy::QuantumGate*){});            // empty deleter
  }
  void setSharedMemUsage(size_t bytes) {
      cuTuple.sharedMemBytes = bytes;
  }
  void setKernelFunction(CUfunction fn) {
      cuTuple.cuFunction = fn;
  }
  CUfunction kernelFunction() const { 
    return cuTuple.cuFunction; 
  }
  size_t sharedMemUsage() const { 
    return cuTuple.sharedMemBytes; 
  }
  void setRegisterUsage(unsigned r) {
    regsPerThread = r; 
  }
  unsigned registerUsage() const { 
    return regsPerThread; 
  }
  #endif
};

struct CUDAKernelGenConfig {
  enum MatrixLoadMode { 
    UseMatImmValues, LoadInDefaultMemSpace, LoadInConstMemSpace
  };
  Precision precision = Precision::F64; // default to double precision
  double zeroTol = 1e-8;
  double oneTol = 1e-8;
  bool forceDenseKernel = false;
  int blockSize = 64; // for now have constant blocksize across kernels
  MatrixLoadMode matrixLoadMode = UseMatImmValues;

  std::ostream& displayInfo(std::ostream& os) const;
};

class CUDAKernelManager : public KernelManagerBase {
  std::vector<CUDAKernelInfo> _cudaKernels;

  enum JITState { JIT_Uninited, JIT_PTXEmitted, JIT_CUFunctionLoaded };
  JITState jitState;
public:
  CUDAKernelManager()
    : KernelManagerBase()
    , _cudaKernels()
    , jitState(JIT_Uninited) {}

  std::vector<CUDAKernelInfo>& kernels() { return _cudaKernels; }
  const std::vector<CUDAKernelInfo>& kernels() const { return _cudaKernels; }

  CUDAKernelManager& genCUDAGate(
      const CUDAKernelGenConfig& config,
      std::shared_ptr<legacy::QuantumGate> gate, const std::string& funcName);
  
  CUDAKernelManager& genCUDAGateMulti(
    const CUDAKernelGenConfig& config,
    const std::vector<std::shared_ptr<legacy::QuantumGate>>& gateList,
    const std::string& funcName);

  CUDAKernelManager& genCUDAGatesFromLegacyCircuitGraph(
      const CUDAKernelGenConfig& config,
      const legacy::CircuitGraph& graph, const std::string& graphName);

  CUDAKernelManager& genCUDAGatesFromLegacyCircuitGraphMulti(
    const CUDAKernelGenConfig& config,
    const legacy::CircuitGraph& graph, const std::string& graphName);

  std::vector<CUDAKernelInfo*>
  collectCUDAKernelsFromLegacyCircuitGraph(const std::string& graphName);

  void emitPTX(
      int nThreads = 1,
      llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O0,
      int verbose = 0);

  std::string getPTXString(int idx) const {
    return std::string(_cudaKernels[idx].ptxString.str());
  }
  
  void dumpPTX(const std::string &kernelName, llvm::raw_ostream &os);

#ifdef CAST_USE_CUDA
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
    for (auto& kernel : _cudaKernels) {
      if (kernel.cuTuple.cuModule) {
        cuModuleUnload(kernel.cuTuple.cuModule);
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

  ///
  void launchCUDAKernel(
      void* dData, int nQubits, CUDAKernelInfo& kernelInfo, int blockSize=64);
  
  void launchCUDAKernelParam(
      void* dData,
      int nQubits,
      CUDAKernelInfo& kernelInfo,
      void* dMatPtr,
      int blockSize=64
  );

#endif // CAST_USE_CUDA
};


} // namespace cast

#endif // CAST_CORE_KERNEL_MANAGER_H
