#include "cuda_jit.hpp"

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Triple.h>

#include <cstring>
#include <mutex>
#include <string>

namespace cast::cuda {

// ── NVPTX target initialisation ─────────────────────────────────────────────

static void ensureNvptxInitialized() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });
}

// ── NVPTX TargetMachine helper ───────────────────────────────────────────────

static llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
createNvptxTargetMachine(uint32_t smMajor, uint32_t smMinor) {
  ensureNvptxInitialized();

  llvm::Triple const triple("nvptx64-nvidia-cuda");
  std::string err;
  const auto *target = llvm::TargetRegistry::lookupTarget(triple, err);
  if (!target)
    return llvm::createStringError("NVPTX target not found: " + err);

  std::string const arch = "sm_" + std::to_string(smMajor) + std::to_string(smMinor);
  llvm::TargetOptions const options;
  auto tm = std::unique_ptr<llvm::TargetMachine>(
      target->createTargetMachine(triple, arch, /*features=*/"", options,
                                  /*reloc=*/std::nullopt));
  if (!tm)
    return llvm::createStringError("failed to create NVPTX TargetMachine for " + arch);
  return tm;
}

// ── RawPwriteStringOstream ────────────────────────────────────────────────
// Writable stream that appends to a std::string; compatible with
// TargetMachine::addPassesToEmitFile which requires raw_pwrite_stream.

namespace {
class RawPwriteStringOstream : public llvm::raw_pwrite_stream {
  std::string &out_;

  void write_impl(const char *Ptr, size_t Size) override { out_.append(Ptr, Size); }
  void pwrite_impl(const char *Ptr, size_t Size, uint64_t Offset) override {
    if (out_.size() < Offset + Size)
      out_.resize(static_cast<size_t>(Offset + Size));
    std::memcpy(out_.data() + Offset, Ptr, Size);
  }
  uint64_t current_pos() const override { return out_.size(); }

public:
  explicit RawPwriteStringOstream(std::string &str) : out_(str) { SetUnbuffered(); }
  ~RawPwriteStringOstream() override { flush(); }
};
} // namespace

// ── optimizeKernelIr ───────────────────────────────────────────────────────

llvm::Error optimizeKernelIr(GeneratedKernel &generated) {
  if (generated.optimized)
    return llvm::Error::success();
  if (!generated.module)
    return llvm::createStringError("kernel module is null");

  if (!generated.tm) {
    auto tm = createNvptxTargetMachine(generated.spec.smMajor, generated.spec.smMinor);
    if (!tm)
      return tm.takeError();
    generated.tm = std::move(*tm);
  }

  llvm::Module &M = *generated.module;

  // Set triple + data layout so NVPTX passes have consistent target info.
  M.setTargetTriple(generated.tm->getTargetTriple());
  M.setDataLayout(generated.tm->createDataLayout());

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PassInstrumentationCallbacks pic;
  llvm::StandardInstrumentations si(M.getContext(), /*DebugLogging=*/false);
  si.registerCallbacks(pic, &mam);

  llvm::PipelineTuningOptions const pto;
  // Pass the NVPTX TM so backend-specific analyses (e.g. TTI) are available.
  llvm::PassBuilder pb(generated.tm.get(), pto, std::nullopt, &pic);
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

  // Cache IR text while we still own the module.
  llvm::raw_string_ostream os(generated.ir);
  M.print(os, /*AAW=*/nullptr);

  generated.optimized = true;
  return llvm::Error::success();
}

// ── compileKernel ───────────────────────────────────────────────────────────

llvm::Error compileKernel(GeneratedKernel &generated, CompiledKernel &out) {
  // Stage 1: optimize (idempotent; also sets triple+layout on module).
  if (auto err = optimizeKernelIr(generated))
    return err;

  // Stage 2: LLVM IR → PTX via NVPTX backend.
  // Reuse the TargetMachine created (or cached) by the optimize step.
  {
    // Verify once more before codegen.
    std::string errStr;
    llvm::raw_string_ostream sstream(errStr);
    if (llvm::verifyModule(*generated.module, &sstream))
      return llvm::createStringError("module verification failed before PTX: " + errStr);

    RawPwriteStringOstream ptxStream(out.ptx);
    llvm::legacy::PassManager pm;
    if (generated.tm->addPassesToEmitFile(pm, ptxStream, /*DwoOut=*/nullptr,
                                          llvm::CodeGenFileType::AssemblyFile))
      return llvm::createStringError("NVPTX TargetMachine cannot emit PTX");
    pm.run(*generated.module);
  }

  if (out.ptx.empty())
    return llvm::createStringError("PTX emission produced empty output");

  // Inject .maxnreg directive between the entry declaration and its body.
  // There is no LLVM API for this; PTX post-processing is the only way.
  // The directive tells the CUDA JIT to allocate up to N physical registers
  // per thread, trading occupancy for less local-memory spilling.
  if (generated.spec.maxNReg > 0) {
    auto pos = out.ptx.find(")\n{");
    if (pos != std::string::npos)
      out.ptx.insert(pos + 1, "\n.maxnreg " + std::to_string(generated.spec.maxNReg));
  }

  out.nGateQubits = generated.nGateQubits;
  out.precision = generated.spec.precision;
  out.funcName = std::move(generated.funcName);
  return llvm::Error::success();
}

} // namespace cast::cuda
