#ifndef CAST_CORE_KERNEL_MANAGER_H
#define CAST_CORE_KERNEL_MANAGER_H

#include "cast/CPU/Config.h" // for cast::get_cpu_num_threads()
#include "utils/TaskDispatcher.h"
#include <llvm/IR/Module.h>
#include <map>
#include <ranges>

namespace cast {

namespace internal {
/// mangled name is formed by 'G' + <length of graphName> + graphName
/// @return mangled name
std::string mangleGraphName(const std::string& graphName);

std::string demangleGraphName(const std::string& mangledName);
} // namespace internal

class KernelManagerBase {
protected:
  // A vector of pairs of LLVM context and module. Expected to be cleared after
  // calling \c initJIT
  utils::TaskDispatcher dispatcher;

  explicit KernelManagerBase(int nWorkerThreads)
      : dispatcher(nWorkerThreads > 0 ? nWorkerThreads
                                      : cast::get_cpu_num_threads()) {}
};

// KernelManager offers CRTP style interface to manage kernel storage and
// provides iterators.
template <typename KernelInfoType>
class KernelManager : public KernelManagerBase {
protected:
  using KernelInfoPtr = std::unique_ptr<KernelInfoType>;

  // A PoolItem is bounded to a KernelInfo, LLVM Context and LLVM Module
  struct PoolItem {
    KernelInfoPtr kernel;
    std::unique_ptr<llvm::LLVMContext> llvmContext;
    std::unique_ptr<llvm::Module> llvmModule;

    PoolItem(const std::string& llvmModuleName) {
      kernel = std::make_unique<KernelInfoType>();
      llvmContext = std::make_unique<llvm::LLVMContext>();
      llvmModule = std::make_unique<llvm::Module>(llvmModuleName, *llvmContext);
    }
  };

  class Pool {
    std::vector<PoolItem> items_;

  public:
    using iterator = typename std::vector<PoolItem>::iterator;
    using const_iterator = typename std::vector<PoolItem>::const_iterator;

    iterator begin() { return items_.begin(); }
    iterator end() { return items_.end(); }
    const_iterator begin() const { return items_.begin(); }
    const_iterator end() const { return items_.end(); }

    std::vector<PoolItem>& items() { return items_; }
    const std::vector<PoolItem>& items() const { return items_; }

    size_t size() const { return items_.size(); }
    bool empty() const { return items_.empty(); }

    void addItem(PoolItem&& item) { items_.emplace_back(std::move(item)); }
  };

  // The storage of kernels. kernelPools_ has a default pool named "_default_".
  using KernelPools = std::map<std::string, Pool>;
  KernelPools kernelPools_;
  constexpr static const char* DEFAULT_POOL_NAME = "_default_";

  /* --- iterator ---*/
public:
  explicit KernelManager(int nWorkerThreads)
      : KernelManagerBase(nWorkerThreads) {
    kernelPools_.insert({DEFAULT_POOL_NAME, Pool()});
  }

  KernelPools& pools() { return kernelPools_; }
  const KernelPools& pools() const { return kernelPools_; }

  Pool& getDefaultPool() { return kernelPools_.at(DEFAULT_POOL_NAME); }

  const Pool& getDefaultPool() const {
    return kernelPools_.at(DEFAULT_POOL_NAME);
  }

  std::span<const PoolItem> getKernelsIn(const std::string& poolName) const {
    auto it = kernelPools_.find(poolName);
    if (it == kernelPools_.end())
      return {}; // empty span
    return std::span<const PoolItem>(it->second.begin(), it->second.end());
  }

  auto all_kernels() {
    return kernelPools_ | std::views::values | std::views::join |
           std::views::transform(&PoolItem::kernel);
  }

  auto all_kernels() const {
    return kernelPools_ | std::views::values | std::views::join |
           std::views::transform(&PoolItem::kernel);
  }

}; // KernelManager

} // namespace cast

#endif // CAST_CORE_KERNEL_MANAGER_H
