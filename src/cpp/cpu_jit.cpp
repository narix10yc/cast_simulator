#include "cpu_jit.h"

#include "cpu_util.h"

#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/TargetSelect.h>

#include <chrono>
#include <cstdlib>
#include <thread>

namespace {

struct CastCpuLaunchArgs {
  void* sv;
  uint64_t ctr_begin;
  uint64_t ctr_end;
  void* p_mat;
};

int default_num_threads() {
  int n_threads = static_cast<int>(std::thread::hardware_concurrency());
  return (n_threads > 0) ? n_threads : 1;
}

void* malloc_matrix_buffer(const CastCpuJittedKernel& kernel) {
  if (kernel.metadata.mode != CAST_CPU_MATRIX_LOAD_STACK_LOAD) {
    return nullptr;
  }

  const uint32_t edge_size = uint32_t(1) << kernel.metadata.n_gate_qubits;
  const size_t n_entries = size_t(edge_size) * size_t(edge_size);
  if (kernel.matrix.size() != n_entries) {
    return nullptr;
  }

  if (kernel.metadata.precision == CAST_CPU_PRECISION_F32) {
    auto* ptr = static_cast<float*>(std::malloc(2 * n_entries * sizeof(float)));
    if (ptr == nullptr) {
      return nullptr;
    }
    for (size_t i = 0; i < n_entries; ++i) {
      ptr[2 * i] = static_cast<float>(kernel.matrix[i].re);
      ptr[2 * i + 1] = static_cast<float>(kernel.matrix[i].im);
    }
    return ptr;
  }

  auto* ptr = static_cast<double*>(std::malloc(2 * n_entries * sizeof(double)));
  if (ptr == nullptr) {
    return nullptr;
  }
  for (size_t i = 0; i < n_entries; ++i) {
    ptr[2 * i] = kernel.matrix[i].re;
    ptr[2 * i + 1] = kernel.matrix[i].im;
  }
  return ptr;
}

} // namespace

llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>>
cast_cpu_jit_create(unsigned n_compile_threads) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  llvm::orc::LLJITBuilder builder;
  builder.setNumCompileThreads((n_compile_threads > 0) ? n_compile_threads : 1);
  auto jit = builder.create();
  if (!jit) {
    return jit.takeError();
  }
  return std::move(*jit);
}

llvm::Error cast_cpu_jit_compile_kernel(llvm::orc::LLJIT& jit,
                                        CastCpuGeneratedKernel& generated,
                                        CastCpuJittedKernel& out) {
  llvm::orc::ThreadSafeModule tsm(std::move(generated.module),
                                  std::move(generated.context));

  tsm.withModuleDo([](llvm::Module& module) {
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    llvm::PassInstrumentationCallbacks pic;
    llvm::StandardInstrumentations si(module.getContext(), false);
    si.registerCallbacks(pic, &mam);

    llvm::PipelineTuningOptions pto;
    // Pass nullptr for the TargetMachine: the kernels are already explicitly
    // vectorized via SIMD shuffle/splat in the IR, so target-specific
    // auto-vectorization passes are not needed here.
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
    mpm.run(module, mam);
  });

  if (auto err = jit.addIRModule(std::move(tsm))) {
    return err;
  }

  auto sym = jit.lookup(generated.func_name);
  if (!sym) {
    return sym.takeError();
  }

  out.metadata = generated.metadata;
  out.func_name = std::move(generated.func_name);
  out.entry = sym->toPtr<cast_cpu_kernel_entry_t>();
  out.matrix = std::move(generated.matrix);
  return llvm::Error::success();
}

llvm::Error cast_cpu_jit_apply_kernel(const CastCpuJittedKernel& kernel,
                                      void* sv,
                                      uint32_t n_qubits_sv,
                                      cast_cpu_precision_t sv_precision,
                                      cast_cpu_simd_width_t sv_simd_width,
                                      int n_threads) {
  if (kernel.entry == nullptr) {
    return llvm::createStringError("kernel executable is not available");
  }
  if (sv == nullptr) {
    return llvm::createStringError("statevector pointer must not be null");
  }
  if (sv_precision != kernel.metadata.precision) {
    return llvm::createStringError(
        "statevector precision does not match kernel");
  }
  if (sv_simd_width != kernel.metadata.simd_width) {
    return llvm::createStringError(
        "statevector SIMD width does not match kernel");
  }

  const unsigned simd_s = cast_cpu_detail::get_simd_s(kernel.metadata.simd_width,
                                                       kernel.metadata.precision);
  const int tmp = static_cast<int>(n_qubits_sv) -
                  static_cast<int>(kernel.metadata.n_gate_qubits) -
                  static_cast<int>(simd_s);
  if (tmp < 0) {
    return llvm::createStringError(
        "statevector has too few qubits for this kernel");
  }

  if (n_threads <= 0) {
    n_threads = default_num_threads();
  }

  const uint64_t n_tasks = uint64_t(1) << tmp;
  // Clamp thread count so every thread gets at least one task.
  // Without this, when n_threads > n_tasks, all work falls to the last thread.
  if (static_cast<uint64_t>(n_threads) > n_tasks) {
    n_threads = static_cast<int>(n_tasks);
  }
  if (n_threads <= 0) {
    n_threads = 1;
  }
  const uint64_t n_tasks_per_thread = n_tasks / static_cast<uint64_t>(n_threads);
  void* p_mat = malloc_matrix_buffer(kernel);
  if (kernel.metadata.mode == CAST_CPU_MATRIX_LOAD_STACK_LOAD &&
      p_mat == nullptr) {
    return llvm::createStringError("failed to allocate matrix buffer");
  }

  std::vector<CastCpuLaunchArgs> launch_args(static_cast<size_t>(n_threads));
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(n_threads));

  for (int i = 0; i < n_threads; ++i) {
    const uint64_t begin = n_tasks_per_thread * uint64_t(i);
    const uint64_t end =
        (i == n_threads - 1) ? n_tasks : (n_tasks_per_thread * uint64_t(i + 1));
    launch_args[static_cast<size_t>(i)] = {sv, begin, end, p_mat};
  }

  for (int i = 0; i < n_threads; ++i) {
    threads.emplace_back(kernel.entry, &launch_args[static_cast<size_t>(i)]);
  }
  for (auto& thread : threads) {
    thread.join();
  }

  std::free(p_mat);
  return llvm::Error::success();
}
