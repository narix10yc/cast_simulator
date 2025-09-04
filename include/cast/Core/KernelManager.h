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

  explicit KernelManagerBase(int nWorkerThreads) : dispatcher(nWorkerThreads) {
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
  }
};

// KernelManager offers CRTP style interface to manage kernel storage and
// provides iterators.
template <typename KernelInfoType>
class KernelManager : public KernelManagerBase {
protected:
  using KernelInfoPtr = std::unique_ptr<KernelInfoType>;
  std::vector<KernelInfoPtr> standaloneKernels_;
  std::map<std::string, std::vector<KernelInfoPtr>> graphKernels_;
  /* --- iterator ---*/
public:
  explicit KernelManager(int nWorkerThreads)
      : KernelManagerBase(nWorkerThreads) {}
      
  class iterator {
    using VecOfKernels = std::vector<KernelInfoPtr>;
    using MapOfKernels = std::map<std::string, VecOfKernels>;
    VecOfKernels::iterator vecIt, vecEnd;
    MapOfKernels::iterator mapIt, mapEnd;

    void proceedToNextMap() {
      while (mapIt != mapEnd) {
        vecIt = mapIt->second.begin();
        vecEnd = mapIt->second.end();
        if (vecIt != vecEnd) // found a non-empty one
          return;
        ++mapIt;
      }
      // reached the end
      vecIt = typename VecOfKernels::iterator();
      vecEnd = typename VecOfKernels::iterator();
    }

  public:
    using value_type = KernelInfoType;
    iterator() = default;

    iterator(VecOfKernels::iterator vecIt,
             VecOfKernels::iterator vecEnd,
             MapOfKernels::iterator mapIt,
             MapOfKernels::iterator mapEnd)
        : vecIt(vecIt), vecEnd(vecEnd), mapIt(mapIt), mapEnd(mapEnd) {
      if (vecIt == vecEnd)
        proceedToNextMap();
    }

    value_type& operator*() const { return *(vecIt->get()); }
    value_type* operator->() const { return &**this; }

    iterator& operator++() {
      ++vecIt;
      if (vecIt == vecEnd)
        proceedToNextMap();

      return *this;
    }

    iterator operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    bool operator==(const iterator& other) const {
      return vecIt == other.vecIt && mapIt == other.mapIt;
    }

    bool operator!=(const iterator& other) const { return !(*this == other); }
  }; // iterator

  iterator begin() {
    return iterator(standaloneKernels_.begin(),
                    standaloneKernels_.end(),
                    graphKernels_.begin(),
                    graphKernels_.end());
  }

  iterator end() {
    return iterator(standaloneKernels_.end(),
                    standaloneKernels_.end(),
                    graphKernels_.end(),
                    graphKernels_.end());
  }

  /* --- const iterator ---*/
public:
  class const_iterator {
    using VecOfKernels = std::vector<KernelInfoPtr>;
    using MapOfKernels = std::map<std::string, VecOfKernels>;
    VecOfKernels::const_iterator vecIt, vecEnd;
    MapOfKernels::const_iterator mapIt, mapEnd;

    void proceedToNextMap() {
      while (mapIt != mapEnd) {
        vecIt = mapIt->second.begin();
        vecEnd = mapIt->second.end();
        if (vecIt != vecEnd) // found a non-empty one
          return;
        ++mapIt;
      }
      // reached the end
      vecIt = typename VecOfKernels::const_iterator();
      vecEnd = typename VecOfKernels::const_iterator();
    }

  public:
    using value_type = const KernelInfoType;
    const_iterator() = default;

    const_iterator(VecOfKernels::const_iterator vecIt,
                   VecOfKernels::const_iterator vecEnd,
                   MapOfKernels::const_iterator mapIt,
                   MapOfKernels::const_iterator mapEnd)
        : vecIt(vecIt), vecEnd(vecEnd), mapIt(mapIt), mapEnd(mapEnd) {
      if (vecIt == vecEnd)
        proceedToNextMap();
    }

    value_type& operator*() const { return *(vecIt->get()); }
    value_type* operator->() const { return &**this; }

    const_iterator& operator++() {
      ++vecIt;
      if (vecIt == vecEnd)
        proceedToNextMap();

      return *this;
    }

    const_iterator operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    bool operator==(const const_iterator& other) const {
      return vecIt == other.vecIt && mapIt == other.mapIt;
    }

    bool operator!=(const const_iterator& other) const {
      return !(*this == other);
    }
  }; // const _iterator

  const_iterator begin() const {
    return const_iterator(standaloneKernels_.begin(),
                          standaloneKernels_.end(),
                          graphKernels_.begin(),
                          graphKernels_.end());
  }

  const_iterator end() const {
    return const_iterator(standaloneKernels_.end(),
                          standaloneKernels_.end(),
                          graphKernels_.end(),
                          graphKernels_.end());
  }
}; // KernelManager

} // namespace cast

#endif // CAST_CORE_KERNEL_MANAGER_H
