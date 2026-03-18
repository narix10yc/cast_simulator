#include "cuda.h"

#include "cuda_gen.h"
#include "cuda_jit.h"
#include "cuda_util.h"

#include <llvm/Support/Error.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using namespace cast_cuda_detail;

// ── Opaque struct bodies ──────────────────────────────────────────────────────

struct cast_cuda_kernel_generator_t {
  std::vector<CastCudaGeneratedKernel> kernels{};
  cast_cuda_kernel_id_t next_kernel_id = 0;
};

// Compilation session stores kernels in insertion order for O(1) indexed access.
struct cast_cuda_kernel_artifacts_t {
  std::vector<CastCudaCompiledKernel> kernels{};
};

// ── Helpers ───────────────────────────────────────────────────────────────────

namespace {

CastCudaGeneratedKernel *find_generated(cast_cuda_kernel_generator_t *gen,
                                        cast_cuda_kernel_id_t kernel_id) {
  if (!gen) return nullptr;
  auto it = std::find_if(gen->kernels.begin(), gen->kernels.end(),
                         [kernel_id](const CastCudaGeneratedKernel &k) {
                           return k.kernel_id == kernel_id;
                         });
  return (it != gen->kernels.end()) ? &*it : nullptr;
}

const CastCudaCompiledKernel *find_compiled(
    const cast_cuda_kernel_artifacts_t *session,
    cast_cuda_kernel_id_t kernel_id) {
  if (!session) return nullptr;
  for (const auto &k : session->kernels)
    if (k.kernel_id == kernel_id) return &k;
  return nullptr;
}

} // namespace

// ── Generator ─────────────────────────────────────────────────────────────────

extern "C" cast_cuda_kernel_generator_t *
cast_cuda_kernel_generator_new(void) {
  try {
    return new cast_cuda_kernel_generator_t();
  } catch (...) {
    return nullptr;
  }
}

extern "C" void
cast_cuda_kernel_generator_delete(cast_cuda_kernel_generator_t *generator) {
  delete generator;
}

extern "C" int
cast_cuda_kernel_generator_generate(cast_cuda_kernel_generator_t *generator,
                                    const cast_cuda_kernel_gen_spec_t *spec,
                                    const cast_cuda_complex64_t *matrix,
                                    size_t matrix_len,
                                    const uint32_t *qubits, size_t n_qubits,
                                    cast_cuda_kernel_id_t *out_kernel_id,
                                    char *err_buf, size_t err_buf_len) {
  if (!generator) {
    write_error_message(err_buf, err_buf_len, "generator must not be null");
    return 1;
  }
  if (!spec) {
    write_error_message(err_buf, err_buf_len, "spec must not be null");
    return 1;
  }
  if (!out_kernel_id) {
    write_error_message(err_buf, err_buf_len, "out_kernel_id must not be null");
    return 1;
  }

  try {
    CastCudaGeneratedKernel kernel;
    kernel.kernel_id     = generator->next_kernel_id++;
    kernel.n_gate_qubits = static_cast<uint32_t>(n_qubits);
    kernel.spec          = *spec;
    kernel.func_name     = "k_" + std::to_string(kernel.kernel_id);
    kernel.context       = std::make_unique<llvm::LLVMContext>();
    kernel.module        = std::make_unique<llvm::Module>(
        kernel.func_name + "_module", *kernel.context);

    auto func = cast_cuda_generate_kernel_ir(
        *spec, matrix, matrix_len, qubits, n_qubits,
        kernel.func_name, *kernel.module);
    if (!func) {
      write_error_message(err_buf, err_buf_len,
                          llvm::toString(func.takeError()));
      return 1;
    }

    *out_kernel_id = kernel.kernel_id;
    generator->kernels.push_back(std::move(kernel));
    clear_error_buffer(err_buf, err_buf_len);
    return 0;
  } catch (const std::exception &ex) {
    write_error_message(err_buf, err_buf_len, ex.what());
    return 1;
  } catch (...) {
    write_error_message(err_buf, err_buf_len,
                        "unknown error in CUDA kernel generation");
    return 1;
  }
}

extern "C" int
cast_cuda_kernel_generator_emit_ir(cast_cuda_kernel_generator_t *generator,
                                   cast_cuda_kernel_id_t kernel_id,
                                   char *out_ir, size_t ir_buf_len,
                                   size_t *out_ir_len,
                                   char *err_buf, size_t err_buf_len) {
  if (!generator) {
    write_error_message(err_buf, err_buf_len, "generator must not be null");
    return 1;
  }

  auto *kernel = find_generated(generator, kernel_id);
  if (!kernel) {
    write_error_message(err_buf, err_buf_len,
                        "kernel id not found in generator");
    return 1;
  }

  if (auto err = cast_cuda_optimize_kernel_ir(*kernel)) {
    write_error_message(err_buf, err_buf_len, llvm::toString(std::move(err)));
    return 1;
  }

  const std::string &ir = kernel->ir;
  if (out_ir_len != nullptr) *out_ir_len = ir.size();

  if (out_ir != nullptr && ir_buf_len > 0) {
    const size_t n = std::min(ir_buf_len - 1, ir.size());
    std::memcpy(out_ir, ir.data(), n);
    out_ir[n] = '\0';
  }

  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

extern "C" int
cast_cuda_kernel_generator_finish(cast_cuda_kernel_generator_t *generator,
                                  cast_cuda_kernel_artifacts_t **out_session,
                                  char *err_buf, size_t err_buf_len) {
  if (!generator) {
    write_error_message(err_buf, err_buf_len, "generator must not be null");
    return 1;
  }
  if (!out_session) {
    write_error_message(err_buf, err_buf_len, "out_session must not be null");
    return 1;
  }

  try {
    auto *session = new cast_cuda_kernel_artifacts_t();
    session->kernels.reserve(generator->kernels.size());

    for (auto &generated : generator->kernels) {
      CastCudaCompiledKernel compiled;
      if (auto err = cast_cuda_compile_kernel(generated, compiled)) {
        write_error_message(
            err_buf, err_buf_len,
            "failed to compile kernel '" + generated.func_name
                + "': " + llvm::toString(std::move(err)));
        delete session;
        return 1;
      }
      session->kernels.push_back(std::move(compiled));
    }

    *out_session = session;
    clear_error_buffer(err_buf, err_buf_len);
    delete generator; // ownership transferred
    return 0;
  } catch (const std::exception &ex) {
    write_error_message(err_buf, err_buf_len, ex.what());
    return 1;
  } catch (...) {
    write_error_message(err_buf, err_buf_len,
                        "unknown error while compiling CUDA kernels");
    return 1;
  }
}

// ── Compilation session — indexed accessors ───────────────────────────────────

extern "C" uint32_t
cast_cuda_kernel_artifacts_n_kernels(
    const cast_cuda_kernel_artifacts_t *session) {
  return session ? static_cast<uint32_t>(session->kernels.size()) : 0u;
}

extern "C" cast_cuda_kernel_id_t
cast_cuda_kernel_artifacts_kernel_id_at(
    const cast_cuda_kernel_artifacts_t *session, uint32_t idx) {
  return (session && idx < session->kernels.size())
      ? session->kernels[idx].kernel_id : 0;
}

extern "C" uint32_t
cast_cuda_kernel_artifacts_n_gate_qubits_at(
    const cast_cuda_kernel_artifacts_t *session, uint32_t idx) {
  return (session && idx < session->kernels.size())
      ? session->kernels[idx].n_gate_qubits : 0;
}

extern "C" uint8_t
cast_cuda_kernel_artifacts_precision_at(
    const cast_cuda_kernel_artifacts_t *session, uint32_t idx) {
  return (session && idx < session->kernels.size())
      ? static_cast<uint8_t>(session->kernels[idx].precision) : 0;
}

extern "C" const char *
cast_cuda_kernel_artifacts_func_name_at(
    const cast_cuda_kernel_artifacts_t *session, uint32_t idx) {
  return (session && idx < session->kernels.size())
      ? session->kernels[idx].func_name.c_str() : nullptr;
}

// ── Compilation session — PTX / cubin extraction ─────────────────────────────

extern "C" int
cast_cuda_kernel_artifacts_emit_ptx(
    cast_cuda_kernel_artifacts_t *session,
    cast_cuda_kernel_id_t kernel_id,
    char *out_ptx, size_t ptx_buf_len, size_t *out_ptx_len,
    char *err_buf, size_t err_buf_len) {
  if (!session) {
    write_error_message(err_buf, err_buf_len, "session must not be null");
    return 1;
  }
  const auto *kernel = find_compiled(session, kernel_id);
  if (!kernel) {
    write_error_message(err_buf, err_buf_len,
                        "kernel id not found in compilation session");
    return 1;
  }

  const std::string &ptx = kernel->ptx;
  if (out_ptx_len != nullptr) *out_ptx_len = ptx.size();

  if (out_ptx != nullptr && ptx_buf_len > 0) {
    const size_t n = std::min(ptx_buf_len - 1, ptx.size());
    std::memcpy(out_ptx, ptx.data(), n);
    out_ptx[n] = '\0';
  }

  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

extern "C" void
cast_cuda_kernel_artifacts_delete(cast_cuda_kernel_artifacts_t *session) {
  delete session;
}
