#include "llvm/Support/TargetSelect.h"

#include "llvm/Object/ObjectFile.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/FormattedStream.h"

#include "cast/CPU/CPUKernelManager.h"
#include "utils/iocolor.h"
#include "utils/TaskDispatcher.h"

#include <cassert>

using namespace cast;
using namespace llvm;

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
    case MatrixLoadMode::UseMatImmValues:
      os << "UseMatImmValues\n"; break;
    case MatrixLoadMode::StackLoadMatElems:
      os << "StackLoadMatElems\n"; break;
    case MatrixLoadMode::StackLoadMatVecs:
      os << "StackLoadMatVecs\n"; break;
  }

  os << CYAN("================================\n");
  return os;
}

std::ostream& CPUKernelManager::displayInfo(std::ostream& os) const {
  int nKernels = _standaloneKernels.size();
  for (const auto& [graphName, kernels] : _graphKernels)
    nKernels += kernels.size();
  os << CYAN("=== CPU Kernel Manager Info ===\n");
  os << "- Is JITed:          " << (isJITed() ? "Yes" : "No") << "\n"
     << "- Number of Kernels: " << nKernels << "\n";

  os << CYAN("=============================\n");
  return os;
}

void CPUKernelManager::ensureAllExecutable(int nThreads, bool progressBar) {
  assert(nThreads > 0);
  if (nThreads == 1) {
    for (auto& kernel : _standaloneKernels)
      ensureExecutable(*kernel);
    for (auto& [graphName, kernels] : _graphKernels) {
      for (auto& kernel : kernels)
        ensureExecutable(*kernel);
    }
  }

  // multi-thread compile
  utils::TaskDispatcher dispatcher(nThreads);
  for (auto& kernel : _standaloneKernels) {
	  dispatcher.enqueue([this, &kernel]() {
      ensureExecutable(*kernel);
	  });
  }
  for (auto& [graphName, kernels] : _graphKernels) {
    for (auto& kernel : kernels) {
      dispatcher.enqueue([this, &kernel]() {
        ensureExecutable(*kernel);
      });
    }
  }
  if (progressBar)
    std::cerr << "Ensure All Executables...\n";
  dispatcher.sync(progressBar);
}

MaybeError<void> CPUKernelManager::initJIT(
    int nThreads, OptimizationLevel optLevel, bool useLazyJIT, int verbose) {
  if (nThreads <= 0) {
    return cast::makeError<void>(
      "Invalid number of threads: " + std::to_string(nThreads));
  }
  if (isJITed()) {
    return cast::makeError<void>("JIT has already been initialized.");
  }

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  applyLLVMOptimization(nThreads, optLevel, /* progressBar */ verbose > 0);

  if (useLazyJIT) {
    // lazy JIT engine
    orc::LLLazyJITBuilder jitBuilder;
    /// It seems not matter the concurrency we set here.
    /// As long as we set it, we can invoke multiple lookup. We control the 
    /// actual number of threads via our custom TaskDispatcher
    jitBuilder.setNumCompileThreads(nThreads);
    auto lazyJIT = cantFail(jitBuilder.create());
    for (auto& [ctx, mod] : llvmContextModulePairs) {
      auto err = lazyJIT->addLazyIRModule(
        orc::ThreadSafeModule(std::move(mod), std::move(ctx)));
      if (err) {
        return cast::makeError<void>(
            "Failed to add lazy IR module: " + llvm::toString(std::move(err)));
      }
    }
    this->llvmJIT = std::move(lazyJIT);
    ensureAllExecutable(nThreads, /* progressBar */ verbose > 0);
  } else {
    // eager JIT engine
    orc::LLJITBuilder eagerJitBuilder;
    eagerJitBuilder.setNumCompileThreads(nThreads);
    auto eagerJIT = cantFail(eagerJitBuilder.create());
    for (auto& [ctx, mod] : llvmContextModulePairs) {
      auto err = eagerJIT->addIRModule(
        orc::ThreadSafeModule(std::move(mod), std::move(ctx)));
      if (err) {
        return cast::makeError<void>(
            "Failed to add IR module: " + llvm::toString(std::move(err)));
      }
    }
    this->llvmJIT = std::move(eagerJIT);
    // eager compile all kernels
    ensureAllExecutable(nThreads, /* progressBar */ verbose > 0);
  }
  this->llvmContextModulePairs.clear();
  return {}; // success
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
  std::cerr << RED("[Err] ") << "In CPUKernelManager::dumpIR: "
            << "Function " << funcName << " not found.\n";
}

#undef DEBUG_TYPE