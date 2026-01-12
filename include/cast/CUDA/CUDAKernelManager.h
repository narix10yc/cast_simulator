/*
 * CUDAKernelManager.h
 */
#ifndef CAST_CUDA_CUDAKERNELMANAGER_H
#define CAST_CUDA_CUDAKERNELMANAGER_H

#include "cast/CPU/Config.h"
#include "cast/CUDA/CUDAJitTls.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "cast/CUDA/Config.h"
#include "cast/Core/IRNode.h"
#include "cast/Core/KernelManager.h"
#include "cast/Core/QuantumGate.h"

#include "utils/InfoLogger.h"
#include "utils/ThreadPool.h"
#include "llvm/IR/LLVMContext.h"

#include <condition_variable>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>

#include <chrono>

#include <cuda.h>
#include <mutex>

namespace cast {

class CUDAKernelManager;

enum class CUDAMatrixLoadMode {
  UseMatImmValues,
  LoadInDefaultMemSpace,
  LoadInConstMemSpace
};

struct CudaKernel {
  std::mutex mtx;
  std::condition_variable cv;

  // LLVMContext is not thread-safe -- for ease of parallel JIT compilation,
  // each kernel has its own LLVM context and module. Each module contains only
  // one function.
  std::unique_ptr<llvm::LLVMContext> llvmContext = nullptr;
  std::unique_ptr<llvm::Module> llvmModule = nullptr;
  llvm::Function* llvmFunc = nullptr;

  // necessary info about this kernel
  ConstQuantumGatePtr gate = nullptr;
  Precision precision = Precision::Unknown;

  // JIT session data
  std::string ptxString;
  std::vector<uint8_t> cubinData;

  // status for lock-free access
  enum Status {
    // not yet generated
    Empty,
    // enqueued for compilation
    Pending,
    // being compiled
    Compiling,
    // finished compilation
    Ready,
  };
  std::atomic<Status> status = Status::Empty;

  /// Note: the caller must ensure the uniqueness of name in the pool
  explicit CudaKernel(const std::string& name) {
    llvmContext = std::make_unique<llvm::LLVMContext>();
    llvmModule = std::make_unique<llvm::Module>(name + "_module", *llvmContext);
  }

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

class CUDAKernelHandler {
  CUDAKernelManager& km;
  CudaKernel* kernel;

public:
  CUDAKernelHandler(CUDAKernelManager& km, CudaKernel* kernel)
      : km(km), kernel(kernel) {}

  void displayInfo(utils::InfoLogger logger) const {
    kernel->displayInfo(logger);
  }
};

/// Manages cuda-related stuff: devices, contexts, streams, modules, etc.
struct CudaCtxManager {

  CudaCtxManager() { CU_CHECK(cuInit(0)); }
};

class CUDAKernelManager {
  using clock_t = std::chrono::steady_clock;
  using time_point_t = clock_t::time_point;

private:
  utils::ThreadPool<CUDAJitTls> tPool;
  CudaCtxManager cuMgr;

  using KernelInfoPtr = std::unique_ptr<CudaKernel>;
  using Pool = std::vector<KernelInfoPtr>;

  std::map<std::string, Pool> kernelPools_;
  constexpr static const char* DEFAULT_POOL_NAME = "_default_";

  void init() {
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    kernelPools_.insert({DEFAULT_POOL_NAME, Pool()});
  }

  // Both gate-sv and superop-dm simulation will boil down to a matrix with
  // target qubits. This function contains the core logics to emit LLVM IR.
  llvm::Expected<llvm::Function*>
  gen_(const CUDAKernelGenConfig& config,
       const ComplexSquareMatrix& matrix,
       const QuantumGate::TargetQubitsType& qubits,
       const std::string& funcName,
       llvm::Module& llvmModule);

  /// An internal function to generate a CUDA kernel and put the kernel in the
  /// specified pool. This function wraps whether gate is a StandardQuantumGate
  /// (with or without noise) or SuperopQuantumGate, and call `gen_` with a
  /// corresponding ComplexSquareMatrix. The generated kernel will be put into
  /// the given pool.
  /// @param funcName: must be unique in the pool as should be guaranteed by the
  /// caller.
  /// @return: the generated CudaKernel.
  llvm::Expected<CudaKernel*> genCUDAGate_(const CUDAKernelGenConfig& config,
                                           ConstQuantumGatePtr gate,
                                           const std::string& funcName,
                                           Pool& pool);

  void enqueueForCompilation(CudaKernel* kernel);

public:
  CUDAKernelManager() : tPool(cast::get_cpu_num_threads()) { init(); }

  CUDAKernelManager(int nWorkerThreads) : tPool(nWorkerThreads) { init(); }

  CUDAKernelManager(const CUDAKernelManager&) = delete;
  CUDAKernelManager(CUDAKernelManager&&) = delete;
  CUDAKernelManager& operator=(const CUDAKernelManager&) = delete;
  CUDAKernelManager& operator=(CUDAKernelManager&&) = delete;

  ~CUDAKernelManager() = default;

  void displayInfo(utils::InfoLogger logger) const;

  Pool& getDefaultPool() { return kernelPools_.at(DEFAULT_POOL_NAME); }

  /// Generate a kernel for a single gate into the default kernel pool.
  /// @param funcName: if empty, a default name "k_<index>" will be assigned. If
  /// provided, it must be unique among all kernels in the default pool.
  /// Unlike `CPUKernelManager`, here the generated kernel will be enqueued into
  /// JIT compilation session. So the returned kernel handler may not be ready
  /// for inspection immediately.
  llvm::Expected<CUDAKernelHandler> genGate(const CUDAKernelGenConfig& config,
                                            ConstQuantumGatePtr gate,
                                            const std::string& funcName = "");

  /// Generate kernels for all gates in the given circuit graph. The generated
  /// kernels will be named as <graphName>_<order>_<gateId>, where order is
  /// the order of the gate in the circuit graph.
  // TODO: do we still need the order
  llvm::Error genGraphGates(const CUDAKernelGenConfig& config,
                            const ir::CircuitGraphNode& graph,
                            const std::string& poolName);

  /* JIT Session */
private:
  static constexpr size_t BUFFER_SIZE = 4;
  struct RingBuffer {
    std::array<CudaKernel*, BUFFER_SIZE> data;

    CudaKernel*& operator[](size_t idx) { return data[idx % BUFFER_SIZE]; }
  };

  RingBuffer buffer_;

public:
  /// Wait for all enqueued kernel compilations to finish.
  void syncCompilation();

  void syncKernelExecution();

}; // class CUDAKernelManager

} // namespace cast

#endif // CAST_CUDA_CUDAKERNELMANAGER_H