/*
 * CUDAKernelManager.h
 */
#ifndef CAST_CUDA_CUDAKERNELMANAGER_H
#define CAST_CUDA_CUDAKERNELMANAGER_H

#include "cast/Core/KernelManager.h"
#include "cast/Core/QuantumGate.h"

#include <cuda.h>
// TODO: We may not need cuda_runtime (also no need to link CUDA::cudart)
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
  CUDAMatrixLoadMode matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;

  std::ostream& displayInfo(std::ostream& os) const;
};

class CUDAKernelManager : public KernelManagerBase {
  std::vector<CUDAKernelInfo> _cudaKernels;

  enum JITState { JIT_Uninited, JIT_PTXEmitted, JIT_CUFunctionLoaded };
  JITState jitState;

public:
  CUDAKernelManager()
      : KernelManagerBase(), _cudaKernels(), jitState(JIT_Uninited) {}

  std::vector<CUDAKernelInfo>& kernels() { return _cudaKernels; }
  const std::vector<CUDAKernelInfo>& kernels() const { return _cudaKernels; }

  CUDAKernelManager& genCUDAGate(const CUDAKernelGenConfig& config,
                                 const QuantumGate* gate,
                                 const std::string& funcName);

  void emitPTX(int nThreads = 1,
               llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O0,
               int verbose = 0);

  std::string getPTXString(int idx) const {
    return std::string(_cudaKernels[idx].ptxString.str());
  }

  void dumpPTX(const std::string& kernelName, llvm::raw_ostream& os);

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
  void launchCUDAKernel(void* dData,
                        int nQubits,
                        CUDAKernelInfo& kernelInfo,
                        int blockSize = 64);

  void launchCUDAKernelParam(void* dData,
                             int nQubits,
                             CUDAKernelInfo& kernelInfo,
                             void* dMatPtr,
                             int blockSize = 64);
};

} // namespace cast

#endif // CAST_CUDA_CUDAKERNELMANAGER_H