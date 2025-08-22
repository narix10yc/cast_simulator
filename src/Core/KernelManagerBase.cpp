#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Passes/PassBuilder.h"

#include "cast/Core/KernelManager.h"

#include "utils/TaskDispatcher.h"
#include "utils/iocolor.h"

#include <cstdlib>
#include "llvm/IR/Verifier.h"
#include "llvm/IR/OptBisect.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"

#include <ranges>

using namespace cast;
using namespace llvm;

std::string cast::internal::mangleGraphName(const std::string& graphName) {
  return "G" + std::to_string(graphName.length()) + graphName;
}

std::string cast::internal::demangleGraphName(const std::string& mangledName) {
  const auto* p = mangledName.data();
  const auto* e = mangledName.data() + mangledName.size();
  assert(p != e);
  assert(*p == 'G' && "Mangled graph name must start with 'G'");
  ++p;
  assert(p != e);
  auto p0 = p;
  while ('0' <= *p && *p <= '9') {
    ++p;
    assert(p != e);
  }
  auto l = std::stoi(std::string(p0, p));
  assert(p + l <= e);
  return std::string(p, p + l);
}

KernelManagerBase::ContextModulePair&
KernelManagerBase::createNewLLVMContextModulePair(const std::string& name) {
  std::lock_guard<std::mutex> lock(mtx);
  auto ctx = std::make_unique<llvm::LLVMContext>();
  llvmContextModulePairs.emplace_back(
      std::move(ctx), std::make_unique<llvm::Module>(name, *ctx));
  return llvmContextModulePairs.back();
}


void KernelManagerBase::applyLLVMOptimization(int nThreads,
                                              OptimizationLevel optLevel,
                                              bool progressBar) {
  assert(nThreads > 0);
  if (optLevel == OptimizationLevel::O0)
    return;

  utils::TaskDispatcher dispatcher(nThreads);

  // Capture Module* by value per task; avoid [&] captures.
  for (auto& [ctxUPtr, modUPtr] : llvmContextModulePairs) {
    llvm::Module* M = modUPtr.get();

    dispatcher.enqueue([M, optLevel]() {
      using namespace llvm;

      // --- Analysis managers (must be constructed in this order) ---
      LoopAnalysisManager LAM;
      FunctionAnalysisManager FAM;
      CGSCCAnalysisManager CGAM;
      ModuleAnalysisManager MAM;

      // --- Pass instrumentation (debug hooks, timers, etc.) ---
      PassInstrumentationCallbacks PIC;
      StandardInstrumentations SI(M->getContext(), /*DebugLogging=*/false);
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
      MPM.addPass(VerifierPass());                          // verify pre
      MPM.addPass(PB.buildPerModuleDefaultPipeline(optLevel));
      MPM.addPass(VerifierPass());                          // verify post

      // Run the pipeline for this module.
      MPM.run(*M, MAM);
    });
  }

  if (progressBar)
    std::cerr << "Applying LLVM Optimization....\n";

  dispatcher.sync(progressBar);
}
