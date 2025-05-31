#include "llvm/Support/TargetSelect.h"

#include "cast/CPU/KernelManagerCPU.h"
#include "utils/iocolor.h"
#include "utils/TaskDispatcher.h"

using namespace cast;
using namespace llvm;

std::ostream& CPUKernelGenConfig::displayInfo(std::ostream& os) const {
  os << std::scientific;
  os << CYAN("=== CPU Kernel Gen Config ===\n")
     << "simd_s:    " << simd_s << "\n"
     << "precision:  " << precision << "\n"
     << "amp format: ";
  switch (this->ampFormat) {
    case AltFormat:
      os << "AltFormat\n"; break;
    case SepFormat:
      os << "SepFormat\n"; break;
    default:
      assert(0 && "Unreachable");
  }

  os << "useFMA     : " << useFMA << "\n"
     << "useFMS     : " << useFMS << "\n"
     << "usePDEP     : " << usePDEP << "\n"
     << "forceDenseKernel : " << forceDenseKernel << "\n"
     << "zeroTolerance : " << zeroTol << "\n"
     << "oneTolerance : " << oneTol << "\n"
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

void CPUKernelManager::ensureAllExecutable(int nThreads, bool progressBar) {
  assert(nThreads > 0);
  if (nThreads == 1) {
    for (auto& kernel : _kernels)
      ensureExecutable(kernel);
    return;
  }

  // multi-thread compile
  utils::TaskDispatcher dispatcher(nThreads);
  for (auto& kernel : _kernels) {
	  dispatcher.enqueue([this, &kernel]() {
      ensureExecutable(kernel);
	  });
  }
  if (progressBar)
    std::cerr << "Ensure All Executables...\n";
  dispatcher.sync(progressBar);
}

void CPUKernelManager::initJIT(
    int nThreads, OptimizationLevel optLevel, bool useLazyJIT, int verbose) {
  assert(nThreads > 0);
  assert(!isJITed() && "Already initialized");

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  applyLLVMOptimization(nThreads, optLevel, /* progressBar */ verbose > 0);

  if (useLazyJIT) {
    // lazy JIT engine
    orc::LLLazyJITBuilder jitBuilder;
    /// It seems not matter how many concurrency we set here.
    /// As long as we set it, we can invoke multiple lookup, and we can 
    /// control the actual number of threads via our custom TaskDispatcher
    jitBuilder.setNumCompileThreads(nThreads);
    auto lazyJIT = cantFail(jitBuilder.create());
    for (auto& [ctx, mod] : llvmContextModulePairs) {
      cantFail(lazyJIT->addLazyIRModule(
        orc::ThreadSafeModule(std::move(mod), std::move(ctx))));
    }
    this->llvmJIT = std::move(lazyJIT);
    ensureAllExecutable(nThreads, /* progressBar */ verbose > 0);
  } else {
    // eager JIT engine
    orc::LLJITBuilder eagerJitBuilder;
    eagerJitBuilder.setNumCompileThreads(nThreads);
    auto eagerJIT = cantFail(eagerJitBuilder.create());
    for (auto& [ctx, mod] : llvmContextModulePairs) {
      cantFail(eagerJIT->addIRModule(
        orc::ThreadSafeModule(std::move(mod), std::move(ctx))));
    }
    this->llvmJIT = std::move(eagerJIT);
    // eager compile all kernels
    ensureAllExecutable(nThreads, /* progressBar */ verbose > 0);
  }
  this->llvmContextModulePairs.clear();
}


#undef DEBUG_TYPE