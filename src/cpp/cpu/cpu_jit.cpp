#include "cpu_jit.hpp"

#include "internal/util.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#include <cstdlib>
#include <cstring>

llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>> cast::cpu::createJit(unsigned nCompileThreads) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  llvm::orc::LLJITBuilder builder;
  builder.setNumCompileThreads((nCompileThreads > 0) ? nCompileThreads : 1);
  auto jit = builder.create();
  if (!jit)
    return jit.takeError();

  return std::move(*jit);
}

llvm::Error cast::cpu::optimizeKernelIr(cast::cpu::GeneratedKernel &generated) {
  if (generated.optimized)
    return llvm::Error::success();
  if (!generated.module)
    return llvm::createStringError("kernel module is null");

  llvm::Module &M = *generated.module;

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PassInstrumentationCallbacks pic;
  llvm::StandardInstrumentations si(M.getContext(), false);
  si.registerCallbacks(pic, &mam);

  llvm::PipelineTuningOptions const pto;
  // Pass nullptr for the TargetMachine: kernels are already explicitly
  // vectorized via SIMD intrinsics, so target auto-vectorization is not needed.
  llvm::PassBuilder pb(nullptr, pto, std::nullopt, &pic);

  pb.registerLoopAnalyses(lam);
  pb.registerFunctionAnalyses(fam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerModuleAnalyses(mam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::ModulePassManager mpm;
  mpm.addPass(llvm::VerifierPass());
  mpm.addPass(pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O1));
  mpm.addPass(llvm::VerifierPass());
  mpm.run(M, mam);

  // Cache the printed IR while we still own the plain Module.
  llvm::raw_string_ostream os(generated.ir);
  M.print(os, /*AAW=*/nullptr);

  generated.optimized = true;
  return llvm::Error::success();
}

llvm::Expected<cast::cpu::CompiledKernelRecord>
cast::cpu::jitCompileKernel(llvm::orc::LLJIT &jit, cast::cpu::GeneratedKernel &generated) {
  // Optimize on the plain Module first so the IR is captured before the Module
  // is moved into the ThreadSafeModule and consumed by the JIT pipeline.
  if (auto err = cast::cpu::optimizeKernelIr(generated))
    return std::move(err);

  // Emit native assembly only when explicitly requested for this kernel.
  std::optional<std::string> asmText;
  if (generated.captureAsm) {
    const llvm::Triple &triple = jit.getTargetTriple();
    std::string errStr;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, errStr);
    if (!target)
      return llvm::createStringError("assembly emission: " + errStr);

    llvm::TargetOptions const options;
    auto tm = std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
        triple, llvm::sys::getHostCPUName().str(), /*features=*/"", options, llvm::Reloc::PIC_));
    if (!tm)
      return llvm::createStringError("failed to create TargetMachine");

    generated.module->setDataLayout(tm->createDataLayout());
    generated.module->setTargetTriple(triple);

    llvm::SmallVector<char, 0> asmBuf;
    llvm::raw_svector_ostream asmOs(asmBuf);
    llvm::legacy::PassManager pm;
    if (tm->addPassesToEmitFile(pm, asmOs, /*DwoOut=*/nullptr, llvm::CodeGenFileType::AssemblyFile))
      return llvm::createStringError("target does not support assembly emission");
    pm.run(*generated.module);

    asmText.emplace(asmBuf.begin(), asmBuf.end());
  }

  llvm::orc::ThreadSafeModule tsm(std::move(generated.module), std::move(generated.context));

  if (auto err = jit.addIRModule(std::move(tsm))) {
    return std::move(err);
  }

  auto sym = jit.lookup(generated.funcName);
  if (!sym)
    return sym.takeError();

  cast::cpu::CompiledKernelRecord out;
  out.metadata = generated.metadata;
  out.entry = sym->toPtr<cast::cpu::KernelEntry>();
  out.matrix = generated.matrix;
  if (generated.captureIr)
    out.irText = std::move(generated.ir);
  out.asmText = std::move(asmText);
  return out;
}
