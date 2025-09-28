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

std::ostream& CPUKernelInfo::displayInfo(std::ostream& os) const {
  os << std::scientific;
  os << CYAN("=== CPU Kernel Info ===\n")
     << "- Precision:       " << static_cast<int>(precision) << "\n"
     << "- LLVM Func Name:  " << llvmFuncName << "\n"
     << "- Matrix Load Mode: ";
  switch (this->matrixLoadMode) {
  case CPUMatrixLoadMode::UseMatImmValues:
    os << "UseMatImmValues\n";
    break;
  case CPUMatrixLoadMode::StackLoadMatElems:
    os << "StackLoadMatElems\n";
    break;
  case CPUMatrixLoadMode::StackLoadMatVecs:
    os << "StackLoadMatVecs\n";
    break;
  }
  os << "- Gate:           " << (void*)(gate.get()) << "\n"
     << "- SIMD Width:     " << static_cast<int>(simdWidth) << "\n"
     << "- Op Count:       " << opCount << "\n"
     << "- Executable:     " << (executable ? "Yes" : "No") << "\n"
     << "- JIT Time:       " << utils::fmt_time(getJitTime()) << "\n"
     << "- Exec Time:      " << utils::fmt_time(getExecTime()) << "\n";

  os << CYAN("=========================\n");
  return os;
}

std::ostream& CPUKernelGenConfig::displayInfo(std::ostream& os) const {
  os << std::scientific;
  os << CYAN("=== CPU Kernel Gen Config ===\n")
     << "SIMD Width:      " << static_cast<int>(simdWidth) << "\n"
     << "Precision:       " << static_cast<int>(precision) << "\n"
     << "Use FMA:         " << useFMA << "\n"
     << "Use FMS:         " << useFMS << "\n"
     << "Use PDEP:        " << usePDEP << "\n"
     << "Zero Tolerance:  " << zeroTol << "\n"
     << "One Tolerance:   " << oneTol << "\n"
     << "matrixLoadMode: ";
  switch (this->matrixLoadMode) {
  case CPUMatrixLoadMode::UseMatImmValues:
    os << "UseMatImmValues\n";
    break;
  case CPUMatrixLoadMode::StackLoadMatElems:
    os << "StackLoadMatElems\n";
    break;
  case CPUMatrixLoadMode::StackLoadMatVecs:
    os << "StackLoadMatVecs\n";
    break;
  }

  os << CYAN("================================\n");
  return os;
}

std::ostream& CPUKernelManager::displayInfo(std::ostream& os) const {
  int nKernels = standaloneKernels_.size();
  for (const auto& [graphName, kernels] : graphKernels_)
    nKernels += kernels.size();
  os << CYAN("=== CPU Kernel Manager Info ===\n");
  os << "- Is JITed:          " << (isJITed() ? "Yes" : "No") << "\n"
     << "- Number of Kernels: " << nKernels << "\n";

  os << CYAN("=============================\n");
  return os;
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