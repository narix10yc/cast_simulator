#ifndef CAST_CPU_KERNEL_MANAGER_CPU_H
#define CAST_CPU_KERNEL_MANAGER_CPU_H

#include "cast/Core/KernelManager.h"
#include "cast/Core/QuantumGate.h"
#include "cast/IR/IRNode.h"

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
  std::shared_ptr<cast::QuantumGate> gate;
  // extra information
  int simd_s;
  int opCount;
};

struct CPUKernelGenConfig {
  enum AmpFormat { AltFormat, SepFormat };

  int simd_s = 2;
  int precision = 64;
  AmpFormat ampFormat = AltFormat;
  bool useFMA = true;
  bool useFMS = true;
  // parallel bits deposit from BMI2
  bool usePDEP = false;
  double zeroTol = 1e-8;
  double oneTol = 1e-8;
  MatrixLoadMode matrixLoadMode = MatrixLoadMode::UseMatImmValues;

  std::ostream& displayInfo(std::ostream& os) const;

  // TODO: set up default configurations
  static const CPUKernelGenConfig NativeDefaultF32;
  static const CPUKernelGenConfig NativeDefaultF64;
};

class CPUKernelManager : public KernelManagerBase {
  std::vector<CPUKernelInfo> _kernels;
  std::unique_ptr<llvm::orc::LLJIT> llvmJIT;

  // Both gate-sv and superop-dm simulation will boil down to a matrix with
  // target qubits. This function contains the core logics to emit LLVM IR.
  llvm::Function* _gen(const CPUKernelGenConfig& config,
                       const ComplexSquareMatrix& matrix,
                       const QuantumGate::TargetQubitsType& qubits,
                       const std::string& funcName);
public:
  CPUKernelManager()
    : KernelManagerBase()
    , _kernels()
    , llvmJIT(nullptr) {}

  std::vector<CPUKernelInfo>& kernels() { return _kernels; }
  const std::vector<CPUKernelInfo>& kernels() const { return _kernels; }

  /// Initialize JIT session. When succeeds, \c llvmContextModulePairs
  /// will be cleared and \c llvmJIT will be non-null. This function can only be
  /// called once and cannot be undone.
  /// \param nThreads number of threads to use.
  /// \param optLevel Apply LLVM optimization passes.
  /// \param useLazyJIT If true, use lazy compilation features provided by LLVM
  /// ORC JIT engine. This means all kernels only get compiled just before being
  /// called. If set to false, all kernels are ready to be executed when this
  /// function returns (good for benchmarks).
  void initJIT(
      int nThreads = 1,
      llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O0,
      bool useLazyJIT = false,
      int verbose = 0);

  bool isJITed() const {
    assert(llvmJIT == nullptr ||
           (llvmContextModulePairs.empty() && llvmJIT != nullptr));
    return llvmJIT != nullptr;
  }

  // This is the main entry point to generate a CPU kernel.
  CPUKernelManager& genCPUGate(const CPUKernelGenConfig& config,
                               QuantumGatePtr gate,
                               const std::string& funcName);

  /// Generate kernels for all gates in the given circuit graph. The generated
  /// kernels will be named as <graphName>_<order>_<gateId>, where order is the
  /// order of the gate in the circuit graph. <order> will be retrieved in
  /// \c collectKernelsFromGraphName 
  CPUKernelManager& genCPUGatesFromGraph(const CPUKernelGenConfig& config,
                                         const ir::CircuitGraphNode& graph,
                                         const std::string& graphName);

  std::vector<CPUKernelInfo*>
  collectKernelsFromGraphName(const std::string& graphName);

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

  void dumpAsm(const std::string& funcName, llvm::raw_ostream& os);

  void ensureAllExecutable(int nThreads = 1, bool progressBar = false);

  void applyCPUKernel(void* sv, int nQubits, CPUKernelInfo& kernelInfo);

  void applyCPUKernel(void* sv, int nQubits, const std::string& funcName);

	// void applyCPUKernel(void* sv, int nQubits, const std::string& funcName, const void* pMatArg);

	// void applyCPUKernel(void* sv, int nQubits, CPUKernelInfo& kernel, const void* pMatArg);

  void applyCPUKernelMultithread(
      void* sv, int nQubits, CPUKernelInfo& kernelInfo, int nThreads);

  void applyCPUKernelMultithread(
      void* sv, int nQubits, const std::string& funcName, int nThreads);
};

};

#endif // CAST_CPU_KERNEL_MANAGER_CPU_H