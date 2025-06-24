#ifndef CAST_CPU_KERNEL_MANAGER_CPU_H
#define CAST_CPU_KERNEL_MANAGER_CPU_H

#include "cast/Core/KernelManager.h"
#include "cast/Core/QuantumGate.h"
#include "cast/IR/IRNode.h"
#include "cast/CPU/Config.h"
#include "utils/MaybeError.h"

#define CPU_KERNEL_TYPE void(void*)

namespace cast {

enum class MatrixLoadMode { 
  // UseMatImmValues: Use immediate values for matrix elements. In the IR, 
  // these elements will be LLVM Constants, which are hardcoded in the
  // assembly.
  UseMatImmValues,
  // StackLoadMatElems: Load the matrix elements into the stack at the very
  // beginning of the kernel.
  StackLoadMatElems,
  // StackLoadMatVecs: Load the matrix elements into the stack at the very
  // beginning of the kernel, but load them as vectors.
  // Notice: This is not in use yet.
  StackLoadMatVecs
};

struct CPUKernelInfo {
  std::function<CPU_KERNEL_TYPE> executable;
  int precision;
  std::string llvmFuncName;
  MatrixLoadMode matrixLoadMode;
  ConstQuantumGatePtr gate;
  // extra information
  CPUSimdWidth simdWidth;
  double opCount;
};

struct CPUKernelGenConfig {
  CPUSimdWidth simdWidth;
  int precision;
  bool useFMA;
  bool useFMS;
  // parallel bits deposit from BMI2
  bool usePDEP;
  double zeroTol;
  double oneTol;
  MatrixLoadMode matrixLoadMode;

  CPUKernelGenConfig()
    : simdWidth(get_cpu_simd_width())
    , precision(64) // default to double precision
    , useFMA(true)
    , useFMS(true)
    , usePDEP(false)
    , zeroTol(1e-8)
    , oneTol(1e-8)
    , matrixLoadMode(MatrixLoadMode::UseMatImmValues) {}

  CPUKernelGenConfig(CPUSimdWidth simdWidth, int precision)
    : simdWidth(simdWidth)
    , precision(precision)
    , useFMA(true)
    , useFMS(true)
    , usePDEP(false)
    , zeroTol(1e-8)
    , oneTol(1e-8)
    , matrixLoadMode(MatrixLoadMode::UseMatImmValues) {}

  int get_simd_s() const {
    return cast::get_simd_s(simdWidth, precision);
  }  

  std::ostream& displayInfo(std::ostream& os) const;
};

class CPUKernelManager : public KernelManagerBase {
  using KernelInfoPtr = std::unique_ptr<CPUKernelInfo>;
  std::vector<KernelInfoPtr> _standaloneKernels;
  std::map<std::string, std::vector<KernelInfoPtr>> _graphKernels;
  std::unique_ptr<llvm::orc::LLJIT> llvmJIT;

  // Both gate-sv and superop-dm simulation will boil down to a matrix with
  // target qubits. This function contains the core logics to emit LLVM IR.
  llvm::Function* _gen(const CPUKernelGenConfig& config,
                       const ComplexSquareMatrix& matrix,
                       const QuantumGate::TargetQubitsType& qubits,
                       const std::string& funcName);

  // Generate a CPU kernel for the given gate. This function will check if the
  // gate is a standard gate or a superoperator gate.
  MaybeError<std::unique_ptr<CPUKernelInfo>> _genCPUGate(
      const CPUKernelGenConfig& config,
      ConstQuantumGatePtr gate,
      const std::string& funcName);

  static std::atomic<int> _standaloneKernelCounter;
public:
  CPUKernelManager()
    : KernelManagerBase()
    , _standaloneKernels()
    , llvmJIT(nullptr) {}

  std::ostream& displayInfo(std::ostream& os) const;

  // Get all standalone kernels.
  const std::vector<KernelInfoPtr>& getAllStandaloneKernels() const {
    return _standaloneKernels;
  }

  // Get kernel by name. Return nullptr if not found.
  const CPUKernelInfo* getKernelByName(const std::string& llvmFuncName) const {
    for (const auto& kernel : _standaloneKernels) {
      if (kernel->llvmFuncName == llvmFuncName)
        return kernel.get();
    }
    for (const auto& [graphName, kernels] : _graphKernels) {
      for (const auto& kernel : kernels) {
        if (kernel->llvmFuncName == llvmFuncName)
          return kernel.get();
      }
    }
    return nullptr;
  }

  /// Initialize JIT session. When succeeds, \c llvmContextModulePairs
  /// will be cleared and \c llvmJIT will be non-null. This function can only be
  /// called once and cannot be undone.
  /// \param nThreads number of threads to use.
  /// \param optLevel Apply LLVM optimization passes.
  /// \param useLazyJIT If true, use lazy compilation features provided by LLVM
  /// ORC JIT engine. This means all kernels only get compiled just before being
  /// called. If set to false, all kernels are ready to be executed when this
  /// function returns (good for benchmarks).
  MaybeError<void> initJIT(
      int nThreads = 1,
      llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O0,
      bool useLazyJIT = false,
      int verbose = 0);

  bool isJITed() const {
    assert(llvmJIT == nullptr ||
           (llvmContextModulePairs.empty() && llvmJIT != nullptr));
    return llvmJIT != nullptr;
  }

  // Generate a CPU kernel for the given gate. The generated kernel will be 
  // put into the standalone kernel pool. This function checks for name 
  // conflicts and will not overwrite existing kernels.
  MaybeError<void> genStandaloneGate(
      const CPUKernelGenConfig& config,
      ConstQuantumGatePtr gate,
      const std::string& funcName);

  /// Generate kernels for all gates in the given circuit graph. The generated
  /// kernels will be named as <graphName>_<order>_<gateId>, where order is the
  /// order of the gate in the circuit graph. <order> will be retrieved in
  /// \c collectKernelsFromGraphName 
  MaybeError<void> genCPUGatesFromGraph(
      const CPUKernelGenConfig& config,
      const ir::CircuitGraphNode& graph,
      const std::string& graphName);

  // std::vector<CPUKernelInfo*>
  // collectKernelsFromGraphName(const std::string& graphName);

  void ensureExecutable(CPUKernelInfo& kernel) {
    // Note: We do not actually need the lock here
    // as it is expected (at least now) each KernelInfo is accesses by a unique
    // thread
    // TODO: we could inline this function into \c initJIT. Maybe introduce a
    // lock inside \c initJIT
    {
      std::lock_guard<std::mutex> lock(mtx);
      if (kernel.executable)
        return;
    }
    auto addr = cantFail(llvmJIT->lookup(kernel.llvmFuncName)).toPtr<CPU_KERNEL_TYPE>();
    // std::cerr << "Kernel " << kernel.llvmFuncName << " addr " << (void*)addr << "\n";
    {
      std::lock_guard<std::mutex> lock(mtx);
      kernel.executable = addr;
    }
  }

	void dumpIR(const std::string& funcName,
              llvm::raw_ostream& os = llvm::errs());

  // TODO: not implemented yet
  void dumpAsm(const std::string& funcName, llvm::raw_ostream& os);

  void ensureAllExecutable(int nThreads = 1, bool progressBar = false);

  MaybeError<void> applyCPUKernel(
      void* sv,
      int nQubits,
      const CPUKernelInfo& kernelInfo,
      int nThreads = 1) const;

  MaybeError<void> applyCPUKernel(
      void* sv,
      int nQubits,
      const std::string& llvmFuncName,
      int nThreads = 1) const;
  
  MaybeError<void> applyCPUKernelsFromGraphMultithread(
      void* sv,
      int nQubits,
      const std::string& graphName,
      int nThreads = 1) const;
};

};

#endif // CAST_CPU_KERNEL_MANAGER_CPU_H