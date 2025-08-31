#ifndef CAST_CORE_KERNEL_MANAGER_H
#define CAST_CORE_KERNEL_MANAGER_H

#include "cast/Core/Precision.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/TargetSelect.h"

#include "utils/TaskDispatcher.h"

namespace cast {

namespace internal {
/// mangled name is formed by 'G' + <length of graphName> + graphName
/// @return mangled name
std::string mangleGraphName(const std::string& graphName);

std::string demangleGraphName(const std::string& mangledName);
} // namespace internal

class KernelManagerBase {
protected:
  struct ContextModulePair {
    std::unique_ptr<llvm::LLVMContext> llvmContext;
    std::unique_ptr<llvm::Module> llvmModule;
  };
  // A vector of pairs of LLVM context and module. Expected to be cleared after
  // calling \c initJIT
  std::vector<ContextModulePair> llvmContextModulePairs;
  // This mutex controls the access to llvmContextModulePairs
  std::mutex mtx;
  utils::TaskDispatcher dispatcher;

  /// A thread-safe version that creates a new llvm Module
  ContextModulePair& createNewLLVMContextModulePair(const std::string& name);

  /// Apply LLVM optimization to all modules inside \c llvmContextModulePairs
  /// As a private member function, this function will be called by \c initJIT
  /// and \c initJITForPTXEmission
  void applyLLVMOptimization(llvm::OptimizationLevel optLevel,
                             bool progressBar);

  explicit KernelManagerBase(int nWorkerThreads)
      : llvmContextModulePairs(), mtx(), dispatcher(nWorkerThreads) {
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
  }
};

} // namespace cast

#endif // CAST_CORE_KERNEL_MANAGER_H
