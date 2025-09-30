#include "llvm/Support/TargetSelect.h"

#include "cast/CPU/CPUKernelManager.h"
#include "utils/Formats.h"
#include "utils/TaskDispatcher.h"
#include "utils/iocolor.h"

#include <cassert>

#define DEBUG_TYPE "cpu-kernel-mgr"
#include "llvm/Support/Debug.h"
// #define LLVM_DEBUG(X) X

using namespace cast;
using namespace llvm;

static const char* matrixLoadModeToStr(CPUMatrixLoadMode mode) {
  switch (mode) {
  case CPUMatrixLoadMode::UseMatImmValues:
    return "UseMatImmValues";
  case CPUMatrixLoadMode::StackLoadMatElems:
    return "StackLoadMatElems";
  case CPUMatrixLoadMode::StackLoadMatVecs:
    return "StackLoadMatVecs";
  default:
    return "Unknown";
  }
}

void CPUKernelInfo::displayInfo(utils::InfoLogger logger) const {
  logger.put("CPUKernelInfo")
      .put("Precision       ", static_cast<int>(precision))
      .put("LLVM Func Name  ", llvmFuncName)
      .put("Matrix Load Mode", matrixLoadModeToStr(matrixLoadMode))
      .put("Gate Ptr        ", (void*)(gate.get()))
      .put("SIMD Width      ", static_cast<int>(simdWidth))
      .put("Op Count        ", opCount)
      .put("Executable      ", (executable ? "Yes" : "No"))
      .put("JIT Time        ", utils::fmt_time(getJitTime()))
      .put("Exec Time       ", utils::fmt_time(getExecTime()));
}

void CPUKernelGenConfig::displayInfo(utils::InfoLogger logger) const {
  logger.put("CPUKernelGenConfig")
      .put("SIMD Width      ", static_cast<int>(simdWidth))
      .put("Precision       ", static_cast<int>(precision))
      .put("Use FMA         ", useFMA)
      .put("Use FMS         ", useFMS)
      .put("Use PDEP        ", usePDEP)
      .put("Zero Tolerance  ", zeroTol)
      .put("One Tolerance   ", oneTol)
      .put("Matrix Load Mode", matrixLoadModeToStr(matrixLoadMode));
}

void CPUKernelManager::displayInfo(utils::InfoLogger logger) const {
  int nKernels = standaloneKernels_.size();
  for (const auto& [graphName, kernels] : graphKernels_)
    nKernels += kernels.size();
  logger.put("CPU Kernel Manager")
      .put("Is JITed         ", isJITed())
      .put("Number of Threads", dispatcher.getNumWorkers())
      .put("Number of Kernels", nKernels);
}

CPUKernelInfo*
CPUKernelManager::getKernelByName(const std::string& llvmFuncName) {
  for (const auto& kernel : standaloneKernels_) {
    if (kernel->llvmFuncName == llvmFuncName)
      return kernel.get();
  }
  for (const auto& [graphName, kernels] : graphKernels_) {
    for (const auto& kernel : kernels) {
      if (kernel->llvmFuncName == llvmFuncName)
        return kernel.get();
    }
  }
  return nullptr;
}

void CPUKernelManager::ensureExecutable(CPUKernelInfo& kernel) {
  // Note: We do not actually need the lock here
  // as it is expected (at least now) each KernelInfo is accesses by a unique
  // thread
  // TODO: we could inline this function into \c initJIT. Maybe introduce a
  // lock inside \c initJIT
  {
    std::lock_guard<std::mutex> lock(mtx);
    if (kernel.executable)
      return;
    kernel.tpJitStart = std::chrono::steady_clock::now();
  }
  auto addr =
      cantFail(llvmJIT->lookup(kernel.llvmFuncName)).toPtr<CPU_KERNEL_TYPE>();
  {
    kernel.tpJitFinish = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mtx);
    kernel.executable = addr;
  }
}

void CPUKernelManager::ensureAllExecutable(bool progressBar) {
  for (auto& kernel : standaloneKernels_) {
    dispatcher.enqueue([this, &kernel]() { ensureExecutable(*kernel); });
  }
  for (auto& [graphName, kernels] : graphKernels_) {
    for (auto& kernel : kernels) {
      dispatcher.enqueue([this, &kernel]() { ensureExecutable(*kernel); });
    }
  }
  if (progressBar)
    std::cerr << "Ensure All Executables...\n";
  dispatcher.sync(progressBar);
}

llvm::Error CPUKernelManager::initJIT(OptimizationLevel optLevel,
                                      bool useLazyJIT,
                                      int verbose) {
  if (isJITed()) {
    return llvm::createStringError("JIT has already been initialized.");
  }

  applyLLVMOptimization(optLevel, /* progressBar */ verbose > 0);

  if (useLazyJIT) {
    // lazy JIT engine
    orc::LLLazyJITBuilder jitBuilder;
    jitBuilder.setNumCompileThreads(dispatcher.getNumWorkers());
    auto lazyJIT = cantFail(jitBuilder.create());
    for (auto& [ctx, mod] : llvmContextModulePairs) {
      if (auto e = lazyJIT->addLazyIRModule(
              orc::ThreadSafeModule(std::move(mod), std::move(ctx)))) {
        return llvm::joinErrors(
            llvm::createStringError("Failed to add lazy IR module"),
            std::move(e));
      }
    }
    this->llvmJIT = std::move(lazyJIT);
    ensureAllExecutable(/* progressBar */ verbose > 0);
  } else {
    // eager JIT engine
    orc::LLJITBuilder eagerJitBuilder;
    eagerJitBuilder.setNumCompileThreads(dispatcher.getNumWorkers());
    auto eagerJIT = cantFail(eagerJitBuilder.create());
    for (auto& [ctx, mod] : llvmContextModulePairs) {
      if (auto e = eagerJIT->addIRModule(
              orc::ThreadSafeModule(std::move(mod), std::move(ctx)))) {
        return llvm::joinErrors(
            llvm::createStringError("Failed to add IR module"), std::move(e));
      }
    }
    this->llvmJIT = std::move(eagerJIT);
    ensureAllExecutable(/* progressBar */ verbose > 0);
  }
  this->llvmContextModulePairs.clear();
  return llvm::Error::success();
}

void CPUKernelManager::dumpIR(const std::string& funcName,
                              llvm::raw_ostream& os) {
  assert(isJITed() == false && "Only supports un-JITed kernels");

  for (const auto& ctxModPair : llvmContextModulePairs) {
    if (auto* func = ctxModPair.llvmModule->getFunction(funcName)) {
      func->print(os, nullptr);
      return;
    }
  }
  std::cerr << RED("[Err] ") << "In CPUKernelManager::dumpIR: " << "Function "
            << funcName << " not found.\n";
}

#undef DEBUG_TYPE