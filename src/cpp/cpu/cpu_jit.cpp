#include "cpu_jit.h"

#include "internal/util.h"

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

llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>> cast_cpu_jit_create(unsigned n_compile_threads) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  llvm::orc::LLJITBuilder builder;
  builder.setNumCompileThreads((n_compile_threads > 0) ? n_compile_threads : 1);
  auto jit = builder.create();
  if (!jit)
    return jit.takeError();

  return std::move(*jit);
}

llvm::Error cast_cpu_optimize_kernel_ir(CastCpuGeneratedKernel &generated) {
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

llvm::Error cast_cpu_jit_compile_kernel(llvm::orc::LLJIT &jit, CastCpuGeneratedKernel &generated,
                                        cast_cpu_jit_kernel_record_t &out) {
  // Optimize on the plain Module first so the IR is captured before the Module
  // is moved into the ThreadSafeModule and consumed by the JIT pipeline.
  if (auto err = cast_cpu_optimize_kernel_ir(generated))
    return err;

  // Emit native assembly only when explicitly requested for this kernel.
  char *asm_ptr = nullptr;
  if (generated.capture_asm) {
    const llvm::Triple &triple = jit.getTargetTriple();
    std::string err_str;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, err_str);
    if (!target)
      return llvm::createStringError("assembly emission: " + err_str);

    llvm::TargetOptions const options;
    auto tm = std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
        triple, llvm::sys::getHostCPUName().str(), /*features=*/"", options, llvm::Reloc::PIC_));
    if (!tm)
      return llvm::createStringError("failed to create TargetMachine");

    generated.module->setDataLayout(tm->createDataLayout());
    generated.module->setTargetTriple(triple);

    llvm::SmallVector<char, 0> asm_buf;
    llvm::raw_svector_ostream asm_os(asm_buf);
    llvm::legacy::PassManager pm;
    if (tm->addPassesToEmitFile(pm, asm_os, /*DwoOut=*/nullptr,
                                llvm::CodeGenFileType::AssemblyFile))
      return llvm::createStringError("target does not support assembly emission");
    pm.run(*generated.module);

    asm_ptr = static_cast<char *>(std::malloc(asm_buf.size() + 1));
    if (!asm_ptr)
      return llvm::createStringError("failed to allocate asm text buffer");
    std::memcpy(asm_ptr, asm_buf.data(), asm_buf.size());
    asm_ptr[asm_buf.size()] = '\0';
  }

  llvm::orc::ThreadSafeModule tsm(std::move(generated.module), std::move(generated.context));

  if (auto err = jit.addIRModule(std::move(tsm))) {
    std::free(asm_ptr);
    return err;
  }

  auto sym = jit.lookup(generated.func_name);
  if (!sym) {
    std::free(asm_ptr);
    return sym.takeError();
  }

  // Copy matrix for StackLoad kernels.
  cast_cpu_complex64_t *matrix_ptr = nullptr;
  size_t const matrix_len = generated.matrix.size();
  if (matrix_len > 0) {
    const size_t nbytes = matrix_len * sizeof(cast_cpu_complex64_t);
    matrix_ptr = static_cast<cast_cpu_complex64_t *>(std::malloc(nbytes));
    if (!matrix_ptr) {
      std::free(asm_ptr);
      return llvm::createStringError("failed to allocate matrix buffer");
    }
    std::memcpy(matrix_ptr, generated.matrix.data(), nbytes);
  }

  out.metadata = generated.metadata;
  out.entry = sym->toPtr<cast_cpu_kernel_entry_t>();
  out.matrix = matrix_ptr;
  out.matrix_len = matrix_len;
  out.asm_text = asm_ptr;
  return llvm::Error::success();
}
