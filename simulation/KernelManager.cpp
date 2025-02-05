#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"

#include "simulation/KernelManager.h"
#include "utils/iocolor.h"

using namespace saot;
using namespace llvm;

void KernelManager::initJIT(OptimizationLevel optLevel) {
  assert(!isJITed() && "Already initialized");

  if (optLevel != OptimizationLevel::O0) {
    // ChatGPT:
    // These must be declared in this order so that they are destroyed in the
    // correct order due to inter-analysis-manager references.
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
    MPM.run(*llvmModule, MAM);
  }

  InitializeNativeTarget();
  InitializeNativeTargetAsmParser();
  InitializeNativeTargetAsmPrinter();

  orc::LLLazyJITBuilder jitBuilder;
  jitBuilder.setNumCompileThreads(std::thread::hardware_concurrency());

  auto lazyJIT = cantFail(jitBuilder.create());
  cantFail(lazyJIT->addLazyIRModule(
    orc::ThreadSafeModule(std::move(llvmModule), std::move(llvmContext))));

  this->llvmContext = nullptr;
  this->llvmModule = nullptr;
  this->llvmJIT = std::move(lazyJIT);
}

std::ostream& CPUKernelGenConfig::displayInfo(std::ostream& os) const {
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
    case UseMatImmValues:
      os << "UseMatImmValues\n"; break;
    case StackLoadMatElems:
      os << "StackLoadMatElems\n"; break;
    case StackLoadMatVecs:
      os << "StackLoadMatVecs\n"; break;
  }

  os << CYAN("================================\n");
  return os;
}