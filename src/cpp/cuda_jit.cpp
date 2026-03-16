#include "cuda_jit.h"

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Triple.h>

#include <nvJitLink.h>

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

static llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
create_nvptx_tm(uint32_t sm_major, uint32_t sm_minor) {
  ensure_nvptx_initialized();

  llvm::Triple triple("nvptx64-nvidia-cuda");
  std::string  err;
  const auto  *target =
      llvm::TargetRegistry::lookupTarget(triple.getTriple(), err);
  if (!target)
    return llvm::createStringError("NVPTX target not found: " + err);

  std::string arch = "sm_" + std::to_string(sm_major) + std::to_string(sm_minor);
  llvm::TargetOptions options;
  auto *tm = target->createTargetMachine(triple, arch, /*features=*/"",
                                         options, /*reloc=*/std::nullopt);
  if (!tm)
    return llvm::createStringError("failed to create NVPTX TargetMachine for "
                                   + arch);
  return std::unique_ptr<llvm::TargetMachine>(tm);
}

// ── raw_pwrite_string_ostream ────────────────────────────────────────────────
// Writable stream that appends to a std::string; compatible with
// TargetMachine::addPassesToEmitFile which requires raw_pwrite_stream.

namespace {
class raw_pwrite_string_ostream : public llvm::raw_pwrite_stream {
  std::string &out_;

  void write_impl(const char *Ptr, size_t Size) override {
    out_.append(Ptr, Size);
  }
  void pwrite_impl(const char *Ptr, size_t Size, uint64_t Offset) override {
    if (out_.size() < Offset + Size)
      out_.resize(static_cast<size_t>(Offset + Size));
    std::memcpy(out_.data() + Offset, Ptr, Size);
  }
  uint64_t current_pos() const override { return out_.size(); }

public:
  explicit raw_pwrite_string_ostream(std::string &str) : out_(str) {
    SetUnbuffered();
  }
  ~raw_pwrite_string_ostream() override { flush(); }
};
} // namespace

// ── cast_cuda_optimize_kernel_ir ─────────────────────────────────────────────

llvm::Error cast_cuda_optimize_kernel_ir(CastCudaGeneratedKernel &generated) {
  if (generated.optimized)
    return llvm::Error::success();
  if (!generated.module)
    return llvm::createStringError("kernel module is null");

  auto tm = create_nvptx_tm(generated.spec.sm_major, generated.spec.sm_minor);
  if (!tm)
    return tm.takeError();

  llvm::Module &M = *generated.module;

  // Set triple + data layout so NVPTX passes have consistent target info.
  M.setTargetTriple((*tm)->getTargetTriple());
  M.setDataLayout((*tm)->createDataLayout());

  llvm::LoopAnalysisManager    lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager    cgam;
  llvm::ModuleAnalysisManager   mam;

  llvm::PassInstrumentationCallbacks pic;
  llvm::StandardInstrumentations     si(M.getContext(), /*DebugLogging=*/false);
  si.registerCallbacks(pic, &mam);

  llvm::PipelineTuningOptions pto;
  // Pass the NVPTX TM so backend-specific analyses (e.g. TTI) are available.
  llvm::PassBuilder pb(tm->get(), pto, std::nullopt, &pic);
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
                                      CastCudaCompiledKernel  &out) {
  // Stage 1: optimize (idempotent; also sets triple+layout on module).
  if (auto err = cast_cuda_optimize_kernel_ir(generated))
    return err;

  // Stage 2: LLVM IR → PTX via NVPTX backend.
  {
    auto tm = create_nvptx_tm(generated.spec.sm_major, generated.spec.sm_minor);
    if (!tm)
      return tm.takeError();

    // Verify once more before codegen.
    std::string errStr;
    llvm::raw_string_ostream sstream(errStr);
    if (llvm::verifyModule(*generated.module, &sstream))
      return llvm::createStringError("module verification failed before PTX: "
                                     + errStr);

    raw_pwrite_string_ostream ptxStream(out.ptx);
    llvm::legacy::PassManager pm;
    if ((*tm)->addPassesToEmitFile(pm, ptxStream, /*DwoOut=*/nullptr,
                                   llvm::CodeGenFileType::AssemblyFile))
      return llvm::createStringError("NVPTX TargetMachine cannot emit PTX");
    pm.run(*generated.module);
  }

  if (out.ptx.empty())
    return llvm::createStringError("PTX emission produced empty output");

  // Stage 3: PTX → cubin via nvJitLink.
  {
    std::string archOpt = "-arch=sm_"
        + std::to_string(10 * generated.spec.sm_major + generated.spec.sm_minor);
    const char *options[] = {archOpt.c_str()};

    nvJitLinkHandle handle = nullptr;
    auto rc = nvJitLinkCreate(&handle, 1, options);
    if (rc != NVJITLINK_SUCCESS) {
      std::string msg = "nvJitLinkCreate failed (rc=" + std::to_string(rc) + ")";
      return llvm::createStringError(msg);
    }
    // RAII: ensure handle is always destroyed.
    struct Guard {
      nvJitLinkHandle h;
      ~Guard() { if (h) nvJitLinkDestroy(&h); }
    } guard{handle};

    rc = nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX,
                          out.ptx.data(), out.ptx.size(),
                          generated.func_name.c_str());
    if (rc != NVJITLINK_SUCCESS) {
      std::string log;
      size_t logSize = 0;
      if (nvJitLinkGetErrorLogSize(handle, &logSize) == NVJITLINK_SUCCESS
          && logSize > 0) {
        log.resize(logSize);
        nvJitLinkGetErrorLog(handle, log.data());
      }
      return llvm::createStringError(
          "nvJitLinkAddData failed (rc=" + std::to_string(rc)
          + (log.empty() ? "" : "): " + log)
          + (log.empty() ? ")" : ""));
    }

    rc = nvJitLinkComplete(handle);
    if (rc != NVJITLINK_SUCCESS) {
      std::string log;
      size_t logSize = 0;
      if (nvJitLinkGetErrorLogSize(handle, &logSize) == NVJITLINK_SUCCESS
          && logSize > 0) {
        log.resize(logSize);
        nvJitLinkGetErrorLog(handle, log.data());
      }
      return llvm::createStringError(
          "nvJitLinkComplete failed (rc=" + std::to_string(rc)
          + (log.empty() ? "" : "): " + log)
          + (log.empty() ? ")" : ""));
    }

    size_t cubinSize = 0;
    nvJitLinkGetLinkedCubinSize(handle, &cubinSize);
    out.cubin.resize(cubinSize);
    nvJitLinkGetLinkedCubin(handle, out.cubin.data());
  }

  out.kernel_id     = generated.kernel_id;
  out.n_gate_qubits = generated.n_gate_qubits;
  out.precision     = generated.spec.precision;
  out.func_name     = std::move(generated.func_name);
  return llvm::Error::success();
}
