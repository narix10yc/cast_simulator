#ifndef CAST_CPU_KERNEL_MANAGER_CPU_H
#define CAST_CPU_KERNEL_MANAGER_CPU_H

#include "cast/CPU/Config.h"
#include "cast/Core/IRNode.h"
#include "cast/Core/KernelManager.h"
#include "cast/Core/QuantumGate.h"
#include "utils/MaybeError.h"

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
  std::function<CPU_KERNEL_TYPE> executable;
  Precision precision;
  std::string llvmFuncName;
  CPUMatrixLoadMode matrixLoadMode;
  ConstQuantumGatePtr gate;
  // extra information
  CPUSimdWidth simdWidth;
  double opCount;
};

struct CPUKernelGenConfig {
  CPUSimdWidth simdWidth;
  Precision precision;
  bool useFMA;
  bool useFMS;
  // parallel bits deposit from BMI2
  bool usePDEP;
  double zeroTol;
  double oneTol;
  CPUMatrixLoadMode matrixLoadMode;

  CPUKernelGenConfig()
      : simdWidth(get_cpu_simd_width()), precision(Precision::F64),
        useFMA(true), useFMS(true), usePDEP(false), zeroTol(1e-8), oneTol(1e-8),
        matrixLoadMode(CPUMatrixLoadMode::UseMatImmValues) {}

  CPUKernelGenConfig(CPUSimdWidth simdWidth, Precision precision)
      : simdWidth(simdWidth), precision(precision), useFMA(true), useFMS(true),
        usePDEP(false), zeroTol(1e-8), oneTol(1e-8),
        matrixLoadMode(CPUMatrixLoadMode::UseMatImmValues) {}

  int get_simd_s() const { return cast::get_simd_s(simdWidth, precision); }

  std::ostream& displayInfo(std::ostream& os) const;
};

class CPUKernelManager : public KernelManagerBase {
  using KernelInfoPtr = std::unique_ptr<CPUKernelInfo>;
  std::vector<KernelInfoPtr> standaloneKernels_;
  std::map<std::string, std::vector<KernelInfoPtr>> graphKernels_;
  std::unique_ptr<llvm::orc::LLJIT> llvmJIT;

  // Both gate-sv and superop-dm simulation will boil down to a matrix with
  // target qubits. This function contains the core logics to emit LLVM IR.
  llvm::Function* gen_(const CPUKernelGenConfig& config,
                       const ComplexSquareMatrix& matrix,
                       const QuantumGate::TargetQubitsType& qubits,
                       const std::string& funcName);

  // Generate a CPU kernel for the given gate. This function will wraps around
  // when gate is a StandardQuantumGate (with or without noise) or
  // SuperopQuantumGate, and call gen_ with a corresponding ComplexSquareMatrix.
  MaybeError<KernelInfoPtr> genCPUGate_(const CPUKernelGenConfig& config,
                                        ConstQuantumGatePtr gate,
                                        const std::string& funcName);

  void ensureExecutable(CPUKernelInfo& kernel);
  void ensureAllExecutable(int nThreads = 1, bool progressBar = false);

public:
  CPUKernelManager()
      : KernelManagerBase(), standaloneKernels_(), llvmJIT(nullptr) {}

  std::ostream& displayInfo(std::ostream& os) const;

  /// Initialize JIT session. When succeeds, \c llvmContextModulePairs
  /// will be cleared and \c llvmJIT will be non-null. This function can only be
  /// called once and cannot be undone.
  /// \param nThreads Number of threads to use.
  /// \param optLevel Apply LLVM optimization passes.
  /// \param useLazyJIT If true, use lazy compilation features provided by LLVM
  /// ORC JIT engine. This means all kernels only get compiled just before being
  /// called. If set to false, all kernels are ready to be executed when this
  /// function returns (good for benchmarks).
  MaybeError<void>
  initJIT(int nThreads = 1,
          llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O0,
          bool useLazyJIT = false,
          int verbose = 0);

  bool isJITed() const {
    assert(llvmJIT == nullptr ||
           (llvmContextModulePairs.empty() && llvmJIT != nullptr));
    return llvmJIT != nullptr;
  }

  /* Generate Kernels */

  /// Generate a CPU kernel for the given gate. The generated kernel will be
  /// put into the standalone kernel pool. This function checks for name
  /// conflicts and will not overwrite existing kernels.
  /// @param gate: The quantum gate. It needs to be in a shared pointer because
  /// the kernel manager keeps track of the gate.
  MaybeError<void> genStandaloneGate(const CPUKernelGenConfig& config,
                                     ConstQuantumGatePtr gate,
                                     const std::string& funcName);

  /// Generate kernels for all gates in the given circuit graph. The generated
  /// kernels will be named as <graphName>_<order>_<gateId>, where order is the
  /// order of the gate in the circuit graph.
  // TODO: do we still need the order
  MaybeError<void> genGraphGates(const CPUKernelGenConfig& config,
                                 const ir::CircuitGraphNode& graph,
                                 const std::string& graphName);

  /* Get Kernels */

  std::span<const KernelInfoPtr> getAllStandaloneKernels() const {
    return std::span<const KernelInfoPtr>(standaloneKernels_);
  }

  // Get kernel by name. Return nullptr if not found.
  const CPUKernelInfo* getKernelByName(const std::string& llvmFuncName) const;

  std::span<const KernelInfoPtr>
  getKernelsFromGraphName(const std::string& graphName) const {
    auto it = graphKernels_.find(graphName);
    if (it == graphKernels_.end())
      return {}; // empty span
    return std::span<const KernelInfoPtr>(it->second);
  }

  /* Apply Kernels */

  MaybeError<void> applyCPUKernel(void* sv,
                                  int nQubits,
                                  const CPUKernelInfo& kernelInfo,
                                  int nThreads = 1) const;

  MaybeError<void> applyCPUKernel(void* sv,
                                  int nQubits,
                                  const std::string& llvmFuncName,
                                  int nThreads = 1) const;

  MaybeError<void> applyCPUKernelsFromGraph(void* sv,
                                            int nQubits,
                                            const std::string& graphName,
                                            int nThreads = 1) const;

  void dumpIR(const std::string& funcName,
              llvm::raw_ostream& os = llvm::errs());

  // TODO: not implemented yet
  void dumpAsm(const std::string& funcName, llvm::raw_ostream& os);

}; // class CPUKernelManager

}; // namespace cast

#endif // CAST_CPU_KERNEL_MANAGER_CPU_H