#include "cast/CUDA/CUDAJitTls.h"
#include "cast/CUDA/Config.h"
#include "utils/iocolor.h"

#include <llvm/MC/TargetRegistry.h>

using namespace cast;

static std::unique_ptr<llvm::TargetMachine> createNVPTXTargetMachine() {
  llvm::Triple triple("nvptx64-nvidia-cuda");
  std::string err;
  const auto* target =
      llvm::TargetRegistry::lookupTarget(triple.getTriple(), err);
  if (target == nullptr) {
    std::cerr << RED("Fatal Err: ") "In looking up NVPTX target: " << err
              << "\n";
    std::abort();
  }

  int major = 0, minor = 0;
  cast::getCudaComputeCapability(major, minor);
  std::string archString =
      "sm_" + std::to_string(major) + std::to_string(minor);
  return std::unique_ptr<llvm::TargetMachine>(
      target->createTargetMachine(triple, archString, "", {}, std::nullopt));
}

CUDAJitTls::CUDAJitTls()
    : targetMachine_(createNVPTXTargetMachine()), PB(targetMachine_.get()) {
  // Register and cross-register
  PB.registerLoopAnalyses(LAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerModuleAnalyses(MAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
}

void CUDAJitTls::runOnModule(llvm::Module& module,
                             llvm::OptimizationLevel optLevel) {
  MPM = PB.buildPerModuleDefaultPipeline(optLevel);
  MPM.run(module, MAM);

  // clean up the analysis managers for next run
  LAM.clear();
  FAM.clear();
  CGAM.clear();
  MAM.clear();
}