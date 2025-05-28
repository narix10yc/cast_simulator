#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Passes/PassBuilder.h"

#include "cast/Core/KernelManager.h"

#include "cast/Fusion.h"

#include "utils/TaskDispatcher.h"
#include "utils/iocolor.h"

#include <ranges>

using namespace cast;
using namespace llvm;

std::string cast::internal::mangleGraphName(const std::string& graphName) {
  return "G" + std::to_string(graphName.length()) + graphName;
}

std::string
cast::internal::demangleGraphName(const std::string& mangledName) {
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
  return std::string(p, p+l);
}

KernelManagerBase::ContextModulePair&
KernelManagerBase::createNewLLVMContextModulePair(const std::string& name) {
  std::lock_guard<std::mutex> lock(mtx);
  auto ctx = std::make_unique<llvm::LLVMContext>();
  llvmContextModulePairs.emplace_back(
    std::move(ctx), std::make_unique<llvm::Module>(name, *ctx));
  return llvmContextModulePairs.back();
}

void KernelManagerBase::applyLLVMOptimization(
    int nThreads, OptimizationLevel optLevel, bool progressBar) {
  assert(nThreads > 0);
  if (optLevel == OptimizationLevel::O0)
    return;

  utils::TaskDispatcher dispatcher(nThreads);
  for (auto& [ctx, mod] : llvmContextModulePairs) {
    // TODO: For some reason, MPM cannot be reused. For now we construct it
    // afresh for every module. Overhead is okay though.
    dispatcher.enqueue([&]() {
      // These must be declared in this order so that they are destroyed in
      // the correct order due to inter-analysis-manager references.
      LoopAnalysisManager LAM;
      FunctionAnalysisManager FAM;
      CGSCCAnalysisManager CGAM;
      ModuleAnalysisManager MAM;

      PassBuilder PB;

      PB.registerLoopAnalyses(LAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.registerModuleAnalyses(MAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
      ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(optLevel);

      MPM.run(*mod, MAM);
    });
  }
  if (progressBar)
    std::cerr << "Applying LLVM Optimization....\n";
  dispatcher.sync(progressBar);
}
