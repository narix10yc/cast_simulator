#ifndef CAST_CPU_KERNEL_MANAGER_CPU_H
#define CAST_CPU_KERNEL_MANAGER_CPU_H

#include "cast/CPU/CPUStatevector.h"
#include "cast/CPU/Config.h"
#include "cast/Core/IRNode.h"
#include "cast/Core/KernelManager.h"
#include "cast/Core/QuantumGate.h"
#include "utils/InfoLogger.h"

#define CPU_KERNEL_TYPE void(void*)

namespace cast {

enum class CPUMatrixLoadMode {
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
  std::function<CPU_KERNEL_TYPE> executable{};
  Precision precision = Precision::Unknown;
  std::string llvmFuncName{};
  CPUMatrixLoadMode matrixLoadMode{};
  ConstQuantumGatePtr gate = nullptr;
  CPUSimdWidth simdWidth = CPUSimdWidth::W_Unknown;
  // we need opCount here due to the zeroTol value
  double opCount;

  using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
  time_point_t tpJitStart{};
  time_point_t tpJitFinish{};
  time_point_t tpExecStart{};
  time_point_t tpExecFinish{};

  CPUKernelInfo() = default;

  void update(Precision precision,
              const std::string& llvmFuncName,
              CPUMatrixLoadMode matrixLoadMode,
              ConstQuantumGatePtr gate,
              CPUSimdWidth simdWidth,
              double opCount) {
    this->precision = precision;
    this->llvmFuncName = llvmFuncName;
    this->matrixLoadMode = matrixLoadMode;
    this->gate = gate;
    this->simdWidth = simdWidth;
    this->opCount = opCount;
  }

  // Get JIT time in seconds
  float getJitTime() const {
    if (tpJitStart.time_since_epoch().count() == 0 ||
        tpJitFinish.time_since_epoch().count() == 0)
      return 0.0f;
    return std::chrono::duration<float>(tpJitFinish - tpJitStart).count();
  }

  // Get execution time in seconds
  float getExecTime() const {
    if (tpExecStart.time_since_epoch().count() == 0 ||
        tpExecFinish.time_since_epoch().count() == 0)
      return 0.0f;
    return std::chrono::duration<float>(tpExecFinish - tpExecStart).count();
  }

  void displayInfo(utils::InfoLogger logger) const;
};

struct CPUKernelGenConfig {
  CPUSimdWidth simdWidth;
  Precision precision;
  bool useFMA = true;
  bool useFMS = true;
  // parallel bits deposit from BMI2
  bool usePDEP = false;
  double zeroTol = 1e-8;
  double oneTol = 1e-8;
  CPUMatrixLoadMode matrixLoadMode = CPUMatrixLoadMode::UseMatImmValues;

  CPUKernelGenConfig(Precision precision)
      : simdWidth(get_cpu_simd_width()), precision(precision) {}

  CPUKernelGenConfig(CPUSimdWidth simdWidth, Precision precision)
      : simdWidth(simdWidth), precision(precision) {}

  int get_simd_s() const { return cast::get_simd_s(simdWidth, precision); }

  void displayInfo(utils::InfoLogger logger) const;
};

class CPUKernelManager : public KernelManager<CPUKernelInfo> {
  std::unique_ptr<llvm::orc::LLJIT> llvmJIT = nullptr;

  // We decide to use LLJIT than LLLazyJIT because multi-threading compilation
  // requies manual control of threads. LLLazyJIT::lookup() does not immediately
  // compile the function. The compilation is only triggered upon calling the
  // function.
  // This is a helper function that creates a LLJIT (stored to llvmJIT). If
  // llvmJIT already exists, this function does nothing.
  llvm::Error initLLVMJIT_();

  // Add a task to compile the given llvmModule. The llvmModule will be
  // transferred to a ThreadSafeModule and stored in the JIT session.
  // This function modifies the given item until item.executable is set.
  llvm::Error compileItem(PoolItem& item, llvm::OptimizationLevel optLevel);

  // Both gate-sv and superop-dm simulation will boil down to a matrix with
  // target qubits. This function contains the core logics to emit LLVM IR.
  // The generated function will be put into the given llvmModule.
  // /c funcName: a unique name for the generated function.
  llvm::Function* gen_(const CPUKernelGenConfig& config,
                       const ComplexSquareMatrix& matrix,
                       const QuantumGate::TargetQubitsType& qubits,
                       const std::string& funcName,
                       llvm::Module& llvmModule);

  // Generate a CPU kernel for the given gate. This function will wraps around
  // when gate is a StandardQuantumGate (with or without noise) or
  // SuperopQuantumGate, and call gen_ with a corresponding ComplexSquareMatrix.
  // The generated kernel will be put into the given pool.
  llvm::Error genCPUGate_(const CPUKernelGenConfig& config,
                          ConstQuantumGatePtr gate,
                          const std::string& funcName,
                          Pool& pool);

public:
  CPUKernelManager(int nWorkerThreads = -1)
      : KernelManager<CPUKernelInfo>(
            nWorkerThreads > 0 ? nWorkerThreads : cast::get_cpu_num_threads()) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeNativeTargetAsmPrinter();

    llvm::cantFail(initLLVMJIT_());
  }

  void displayInfo(utils::InfoLogger logger) const;

  llvm::Error
  compilePool(const std::string& poolName,
              llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O1,
              bool progressBar = false);

  llvm::Error compileDefaultPool(
      llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O1,
      bool progressBar = false) {
    return compilePool(DEFAULT_POOL_NAME, optLevel, progressBar);
  }

  /// Initialize JIT session. When succeeds, \c llvmContextModulePairs
  /// will be cleared and \c llvmJIT will be non-null. This function can only be
  /// called once and cannot be undone.
  /// \param nThreads Number of threads to use.
  /// \param optLevel Apply LLVM optimization passes.
  /// \param useLazyJIT If true, use lazy compilation features provided by LLVM
  /// ORC JIT engine. This means all kernels only get compiled just before being
  /// called. If set to false, all kernels are ready to be executed when this
  /// function returns (good for benchmarks).
  llvm::Error
  compileAll(llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O1,
             bool progressBar = false);

  /* Generate Kernels */

  /// Generate a CPU kernel for the given gate. The generated kernel will be
  /// put into the default kernel pool. This function checks for name
  /// conflicts and will not overwrite existing kernels.
  /// @param gate: The quantum gate. It needs to be in a shared pointer because
  /// the kernel manager keeps track of the gate.
  llvm::Error genGate(const CPUKernelGenConfig& config,
                      ConstQuantumGatePtr gate,
                      const std::string& funcName);

  /// Generate kernels for all gates in the given circuit graph. The generated
  /// kernels will be named as <graphName>_<order>_<gateId>, where order is the
  /// order of the gate in the circuit graph.
  // TODO: do we still need the order
  llvm::Error genGraphGates(const CPUKernelGenConfig& config,
                            const ir::CircuitGraphNode& graph,
                            const std::string& graphName);

  /* Get Kernels */

  // Get kernel by name. Return nullptr if not found.
  CPUKernelInfo* getKernelByName(const std::string& llvmFuncName);

  std::span<const PoolItem> getPool(const std::string& poolName) const {
    auto it = kernelPools_.find(poolName);
    if (it == kernelPools_.end())
      return {}; // empty span
    return std::span<const PoolItem>(it->second);
  }

  /* Apply Kernels */

  llvm::Error applyCPUKernel(void* sv,
                             int nQubitsSV,
                             CPUKernelInfo& kernelInfo,
                             int nThreads = 1);

  llvm::Error applyCPUKernel(void* sv,
                             int nQubitsSV,
                             const std::string& llvmFuncName,
                             int nThreads = 1);

  llvm::Error applyCPUKernel(CPUStatevectorF32& sv,
                             CPUKernelInfo& kernelInfo,
                             int nThreads = 1) {
    return applyCPUKernel(sv.data(), sv.nQubits(), kernelInfo, nThreads);
  }

  llvm::Error applyCPUKernel(CPUStatevectorF64& sv,
                             CPUKernelInfo& kernelInfo,
                             int nThreads = 1) {
    return applyCPUKernel(sv.data(), sv.nQubits(), kernelInfo, nThreads);
  }

  llvm::Error applyCPUKernel(CPUStatevectorF32& sv,
                             const std::string& llvmFuncName,
                             int nThreads = 1) {
    return applyCPUKernel(sv.data(), sv.nQubits(), llvmFuncName, nThreads);
  }

  llvm::Error applyCPUKernel(CPUStatevectorF64& sv,
                             const std::string& llvmFuncName,
                             int nThreads = 1) {
    return applyCPUKernel(sv.data(), sv.nQubits(), llvmFuncName, nThreads);
  }

  llvm::Error applyCPUKernelsFromGraph(void* sv,
                                       int nQubitsSV,
                                       const std::string& graphName,
                                       int nThreads = 1);

  void dumpIR(const std::string& funcName,
              llvm::raw_ostream& os = llvm::errs());

  // TODO: not implemented yet
  void dumpAsm(const std::string& funcName, llvm::raw_ostream& os);

}; // class CPUKernelManager
}; // namespace cast

#endif // CAST_CPU_KERNEL_MANAGER_CPU_H