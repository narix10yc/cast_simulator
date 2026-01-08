#include "cast/CPU/CPUKernelManager.h"
#include "utils/Formats.h"
#include "utils/ThreadPool.h"

#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>

#define DEBUG_TYPE "cpu-kernel-mgr"
#include <llvm/Support/Debug.h>
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
  int nKernels = 0;
  for (const auto& [_, poolValue] : kernelPools_)
    nKernels += poolValue.size();
  logger.put("CPU Kernel Manager")
      .put("Num of Threads", tPool.getNumWorkers())
      .put("Num of Kernels", nKernels);
}

CPUKernelInfo*
CPUKernelManager::getKernelByName(const std::string& llvmFuncName) {
  for (const auto& kernel : all_kernels()) {
    assert(kernel != nullptr);
    if (kernel->llvmFuncName == llvmFuncName)
      return kernel.get();
  }
  return nullptr;
}

llvm::Error CPUKernelManager::compileItem(PoolItem& item,
                                          llvm::OptimizationLevel optLevel) {
  orc::ThreadSafeModule TSM(std::move(item.llvmModule),
                            std::move(item.llvmContext));

  TSM.withModuleDo([optLevel](llvm::Module& M) {
    // --- Analysis managers (must be constructed in this order) ---
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    // --- Pass instrumentation (debug hooks, timers, etc.) ---
    PassInstrumentationCallbacks PIC;
    StandardInstrumentations SI(M.getContext(), /*DebugLogging=*/false);
    SI.registerCallbacks(PIC, &MAM);

    // Use the PassBuilder ctor that takes PIC (portable for LLVM 20.x).
    PipelineTuningOptions PTO;
    PassBuilder PB(/*TM=*/nullptr, PTO, /*PGO=*/std::nullopt, &PIC);

    // Register analyses and cross-proxies.
    PB.registerLoopAnalyses(LAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerModuleAnalyses(MAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Wrap default pipeline with verifiers (pre & post).
    ModulePassManager MPM;
    MPM.addPass(VerifierPass()); // verify pre
    MPM.addPass(PB.buildPerModuleDefaultPipeline(optLevel));
    MPM.addPass(VerifierPass()); // verify post

    // Run the pipeline for this module.
    MPM.run(M, MAM);
  });

  if (auto e = llvmJIT->addIRModule(std::move(TSM))) {
    return llvm::joinErrors(llvm::createStringError("Failed to add IR module"),
                            std::move(e));
  }

  // currently the JIT time only records the lookup time.

  item.kernel->tpJitStart = std::chrono::steady_clock::now();
  auto addr = llvmJIT->lookup(item.kernel->llvmFuncName);
  if (!addr) {
    return llvm::joinErrors(
        llvm::createStringError("Failed to lookup function " +
                                item.kernel->llvmFuncName),
        addr.takeError());
  }
  item.kernel->tpJitFinish = std::chrono::steady_clock::now();

  item.kernel->executable = addr->toPtr<CPU_KERNEL_TYPE>();

  return llvm::Error::success();
}

llvm::Error CPUKernelManager::initLLVMJIT_() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  if (llvmJIT != nullptr)
    return llvm::Error::success();

  // eager JIT engine
  orc::LLJITBuilder eagerJitBuilder;
  eagerJitBuilder.setNumCompileThreads(tPool.getNumWorkers());
  auto tmp = eagerJitBuilder.create();
  if (!tmp) {
    return llvm::joinErrors(llvm::createStringError("Failed to create LLJIT"),
                            tmp.takeError());
  }
  llvmJIT = std::move(*tmp);
  return llvm::Error::success();
}

llvm::Error CPUKernelManager::compilePool(const std::string& poolName,
                                          llvm::OptimizationLevel optLevel,
                                          bool progressBar) {
  assert(llvmJIT != nullptr && "llvmJIT is null");

  auto it = kernelPools_.find(poolName);
  if (it == kernelPools_.end())
    return llvm::createStringError("Pool " + poolName + " not found");

  for (auto& item : it->second) {
    tPool.enqueueMayErr([this, &item, optLevel]() -> llvm::Error {
      if (item.kernel->executable)
        return llvm::Error::success();
      return compileItem(item, optLevel);
    });
  }

  tPool.sync(progressBar);
  return tPool.takeError();
}

llvm::Error CPUKernelManager::compileAllPools(OptimizationLevel optLevel,
                                              bool progressBar) {
  assert(llvmJIT != nullptr && "llvmJIT is null");
  llvm::Error err = llvm::Error::success();
  for (auto& [name, pool] : kernelPools_)
    err = llvm::joinErrors(compilePool(name), std::move(err));

  return err;
}

llvm::Expected<std::string>
CPUKernelManager::getIR(const std::string& funcName) const {
  // search in all pools
  // If JIT is requested we cannot retrieve the IR
  for (const auto& [_, pool] : kernelPools_) {
    for (const auto& item : pool) {
      if (item.kernel->llvmFuncName != funcName)
        continue;
      if (item.llvmModule == nullptr)
        continue;
      // got it there
      for (const auto& f : item.llvmModule->functions()) {
        if (f.getName() == funcName) {
          std::string str;
          llvm::raw_string_ostream os(str);
          f.print(os);
          return str;
        }
      }

      return llvm::createStringError("We found the module, but function '" +
                                     funcName + "' is not in the module");
    }
  }

  if (llvmJIT == nullptr)
    return llvm::createStringError("Cannot find kernel '" + funcName + "'");

  return llvm::createStringError(
      "JIT'ed session does not support retrieving IR");
}

#undef DEBUG_TYPE