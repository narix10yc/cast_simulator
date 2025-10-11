#ifndef CAST_CORE_KERNEL_MANAGER_H
#define CAST_CORE_KERNEL_MANAGER_H

#include "cast/Core/Precision.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/TargetSelect.h"

#include "utils/TaskDispatcher.h"

#include <ranges>

namespace cast {

namespace internal {
/// mangled name is formed by 'G' + <length of graphName> + graphName
/// @return mangled name
std::string mangleGraphName(const std::string& graphName);

std::string demangleGraphName(const std::string& mangledName);
} // namespace internal

class KernelManagerBase {
private:
  // This mutex controls the access to llvmContextModulePairs
  std::mutex mtx_;

protected:
  // A vector of pairs of LLVM context and module. Expected to be cleared after
  // calling \c initJIT
  utils::TaskDispatcher dispatcher;

  /// A thread-safe version that creates a new llvm Module
  llvm::Module* createNewLLVMContextModulePair(const std::string& name);

  /// Apply LLVM optimization to all modules inside \c llvmContextModulePairs
  /// As a protected member function, this function will be called by \c initJIT
  /// and \c initJITForPTXEmission
  void applyLLVMOptimization(llvm::OptimizationLevel optLevel,
                             bool progressBar);

  explicit KernelManagerBase(int nWorkerThreads) : dispatcher(nWorkerThreads) {}
};

// KernelManager offers CRTP style interface to manage kernel storage and
// provides iterators.
template <typename KernelInfoType>
class KernelManager : public KernelManagerBase {
protected:
  using KernelInfoPtr = std::unique_ptr<KernelInfoType>;

  struct Item {
    KernelInfoPtr kernel;
    std::unique_ptr<llvm::LLVMContext> llvmContext;
    std::unique_ptr<llvm::Module> llvmModule;

    Item(const std::string& llvmModuleName) {
      kernel = std::make_unique<KernelInfoType>();
      llvmContext = std::make_unique<llvm::LLVMContext>();
      llvmModule = std::make_unique<llvm::Module>(llvmModuleName, *llvmContext);
    }
  };

  // The storage of kernels. kernelPools_ has a default pool named "_default_".
  using KernelPool = std::map<std::string, std::vector<Item>>;
  KernelPool kernelPools_;
  constexpr static const char* DEFAULT_POOL_NAME = "_default_";

  /* --- iterator ---*/
public:
  explicit KernelManager(int nWorkerThreads)
      : KernelManagerBase(nWorkerThreads) {
    kernelPools_.insert({DEFAULT_POOL_NAME, std::vector<Item>()});
  }

  auto all_kernels() {
    return kernelPools_ | std::views::values | std::views::join |
           std::views::transform(&Item::kernel);
  }

  auto all_kernels() const {
    return kernelPools_ | std::views::values | std::views::join |
           std::views::transform(&Item::kernel);
  }

}; // KernelManager

} // namespace cast

#endif // CAST_CORE_KERNEL_MANAGER_H
