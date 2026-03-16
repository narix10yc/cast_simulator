#include "cuda.h"

#include "cuda_gen.h"
#include "cuda_jit.h"
#include "cuda_util.h"

#include <llvm/Support/Error.h>

#include <exception>
#include <memory>
#include <string>
#include <vector>

using namespace cast_cuda_detail;

// ── Opaque struct body (generator only; session body lives in cuda_jit.h) ────

struct cast_cuda_kernel_generator_t {
  std::vector<CastCudaGeneratedKernel> kernels{};
  cast_cuda_kernel_id_t next_kernel_id = 0;
};

// ── Helpers ──────────────────────────────────────────────────────────────────

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
    const cast_cuda_compilation_session_t *session,
    cast_cuda_kernel_id_t kernel_id) {
  if (!session) return nullptr;
  auto it = session->kernels.find(kernel_id);
  return (it != session->kernels.end()) ? &it->second : nullptr;
}

} // namespace

// ── Generator ────────────────────────────────────────────────────────────────

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
    kernel.kernel_id    = generator->next_kernel_id++;
    kernel.n_gate_qubits = static_cast<uint32_t>(n_qubits);
    kernel.spec         = *spec;
    kernel.func_name    = "k_" + std::to_string(kernel.kernel_id);
    kernel.context   = std::make_unique<llvm::LLVMContext>();
    kernel.module    = std::make_unique<llvm::Module>(
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

  // Optimize (idempotent); populates kernel->ir.
  if (auto err = cast_cuda_optimize_kernel_ir(*kernel)) {
    write_error_message(err_buf, err_buf_len, llvm::toString(std::move(err)));
    return 1;
  }

  const std::string &ir = kernel->ir;
  if (out_ir_len != nullptr)
    *out_ir_len = ir.size();

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
                                  cast_cuda_compilation_session_t **out_session,
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
    auto *session = new cast_cuda_compilation_session_t();
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
      const auto kid = compiled.kernel_id;
      session->kernels.emplace(kid, std::move(compiled));
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

// ── Compilation session ───────────────────────────────────────────────────────

extern "C" int
cast_cuda_compilation_session_emit_ptx(
    cast_cuda_compilation_session_t *session,
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
  if (out_ptx_len != nullptr)
    *out_ptx_len = ptx.size();

  if (out_ptx != nullptr && ptx_buf_len > 0) {
    const size_t n = std::min(ptx_buf_len - 1, ptx.size());
    std::memcpy(out_ptx, ptx.data(), n);
    out_ptx[n] = '\0';
  }

  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

extern "C" int
cast_cuda_compilation_session_emit_cubin(
    cast_cuda_compilation_session_t *session,
    cast_cuda_kernel_id_t kernel_id,
    uint8_t *out_cubin, size_t cubin_buf_len, size_t *out_cubin_len,
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

  const auto &cubin = kernel->cubin;
  if (out_cubin_len != nullptr)
    *out_cubin_len = cubin.size();

  if (out_cubin != nullptr && cubin_buf_len > 0) {
    const size_t n = std::min(cubin_buf_len, cubin.size());
    std::memcpy(out_cubin, cubin.data(), n);
  }

  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

extern "C" void
cast_cuda_compilation_session_delete(cast_cuda_compilation_session_t *session) {
  delete session;
}
