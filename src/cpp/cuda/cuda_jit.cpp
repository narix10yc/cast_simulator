#include "cuda_jit.h"

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

// ── NVPTX target initialisation ─────────────────────────────────────────────

static void ensure_nvptx_initialized() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });
}

// ── NVPTX TargetMachine helper ───────────────────────────────────────────────

static llvm::Expected<std::unique_ptr<llvm::TargetMachine>> create_nvptx_tm(uint32_t sm_major,
                                                                            uint32_t sm_minor) {
  ensure_nvptx_initialized();

  llvm::Triple triple("nvptx64-nvidia-cuda");
  std::string err;
  const auto *target = llvm::TargetRegistry::lookupTarget(triple, err);
  if (!target)
    return llvm::createStringError("NVPTX target not found: " + err);

  std::string arch = "sm_" + std::to_string(sm_major) + std::to_string(sm_minor);
  llvm::TargetOptions options;
  auto tm = std::unique_ptr<llvm::TargetMachine>(
      target->createTargetMachine(triple, arch, /*features=*/"", options,
                                  /*reloc=*/std::nullopt));
  if (!tm)
    return llvm::createStringError("failed to create NVPTX TargetMachine for " + arch);
  return tm;
}

// ── raw_pwrite_string_ostream ────────────────────────────────────────────────
// Writable stream that appends to a std::string; compatible with
// TargetMachine::addPassesToEmitFile which requires raw_pwrite_stream.

namespace {
class raw_pwrite_string_ostream : public llvm::raw_pwrite_stream {
  std::string &out_;

  void write_impl(const char *Ptr, size_t Size) override { out_.append(Ptr, Size); }
  void pwrite_impl(const char *Ptr, size_t Size, uint64_t Offset) override {
    if (out_.size() < Offset + Size)
      out_.resize(static_cast<size_t>(Offset + Size));
    std::memcpy(out_.data() + Offset, Ptr, Size);
  }
  uint64_t current_pos() const override { return out_.size(); }

public:
  explicit raw_pwrite_string_ostream(std::string &str) : out_(str) { SetUnbuffered(); }
  ~raw_pwrite_string_ostream() override { flush(); }
};
} // namespace

// ── cast_cuda_optimize_kernel_ir ─────────────────────────────────────────────

llvm::Error cast_cuda_optimize_kernel_ir(CastCudaGeneratedKernel &generated) {
  if (generated.optimized)
    return llvm::Error::success();
  if (!generated.module)
    return llvm::createStringError("kernel module is null");

  if (!generated.tm) {
    auto tm = create_nvptx_tm(generated.spec.sm_major, generated.spec.sm_minor);
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

  llvm::PipelineTuningOptions pto;
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

// ── cast_cuda_compile_kernel ─────────────────────────────────────────────────

llvm::Error cast_cuda_compile_kernel(CastCudaGeneratedKernel &generated,
                                     CastCudaCompiledKernel &out) {
  // Stage 1: optimize (idempotent; also sets triple+layout on module).
  if (auto err = cast_cuda_optimize_kernel_ir(generated))
    return err;

  // Stage 2: LLVM IR → PTX via NVPTX backend.
  // Reuse the TargetMachine created (or cached) by the optimize step.
  {
    // Verify once more before codegen.
    std::string errStr;
    llvm::raw_string_ostream sstream(errStr);
    if (llvm::verifyModule(*generated.module, &sstream))
      return llvm::createStringError("module verification failed before PTX: " + errStr);

    raw_pwrite_string_ostream ptxStream(out.ptx);
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
  if (generated.spec.maxnreg > 0) {
    auto pos = out.ptx.find(")\n{");
    if (pos != std::string::npos)
      out.ptx.insert(pos + 1, "\n.maxnreg " + std::to_string(generated.spec.maxnreg));
  }

  out.n_gate_qubits = generated.n_gate_qubits;
  out.precision = generated.spec.precision;
  out.func_name = std::move(generated.func_name);
  return llvm::Error::success();
}
