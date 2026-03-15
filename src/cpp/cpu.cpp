#include "cpu.h"

#include "cpu_gen.h"
#include "cpu_jit.h"

#include <llvm/Support/Error.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

void write_error_message(char *err_buf, size_t err_buf_len, const std::string &msg) {
  if (err_buf == nullptr || err_buf_len == 0) {
    return;
  }
  const size_t n = std::min(err_buf_len - 1, msg.size());
  std::memcpy(err_buf, msg.data(), n);
  err_buf[n] = '\0';
}

void clear_error_buffer(char *err_buf, size_t err_buf_len) {
  if (err_buf != nullptr && err_buf_len > 0) {
    err_buf[0] = '\0';
  }
}

} // namespace

struct cast_cpu_kernel_generator_t {
  std::vector<CastCpuGeneratedKernel> kernels{};
  cast_cpu_kernel_id_t next_kernel_id = 0;
};

struct cast_cpu_jit_session_t {
  std::unique_ptr<llvm::orc::LLJIT> jit{};
  // Keyed by kernel_id for O(1) lookup.
  std::unordered_map<cast_cpu_kernel_id_t, CastCpuJittedKernel> kernels{};
};

namespace {

const CastCpuJittedKernel *find_jitted_kernel(const cast_cpu_jit_session_t *session,
                                              cast_cpu_kernel_id_t kernel_id) {
  if (session == nullptr) {
    return nullptr;
  }
  auto it = session->kernels.find(kernel_id);
  return (it != session->kernels.end()) ? &it->second : nullptr;
}

} // namespace

extern "C" cast_cpu_kernel_generator_t *cast_cpu_kernel_generator_new(void) {
  try {
    return new cast_cpu_kernel_generator_t();
  } catch (...) {
    return nullptr;
  }
}

extern "C" void cast_cpu_kernel_generator_delete(cast_cpu_kernel_generator_t *generator) {
  delete generator;
}

extern "C" int cast_cpu_kernel_generator_generate(
    cast_cpu_kernel_generator_t *generator, const cast_cpu_kernel_gen_spec_t *spec,
    const cast_cpu_complex64_t *matrix, size_t matrix_len, const uint32_t *qubits, size_t n_qubits,
    cast_cpu_kernel_id_t *out_kernel_id, char *err_buf, size_t err_buf_len) {
  // Null-pointer guards for arguments that cast_cpu_generate_kernel_ir cannot
  // check on its own (it receives spec by reference and writes to out_kernel_id
  // only after the call succeeds).
  if (generator == nullptr) {
    write_error_message(err_buf, err_buf_len, "generator must not be null");
    return 1;
  }
  if (spec == nullptr) {
    write_error_message(err_buf, err_buf_len, "spec must not be null");
    return 1;
  }
  if (out_kernel_id == nullptr) {
    write_error_message(err_buf, err_buf_len, "out_kernel_id must not be null");
    return 1;
  }

  // All other validation (precision, simd_width, mode, tolerances, matrix
  // pointer, qubits pointer, matrix length, qubit ordering) is performed
  // inside cast_cpu_generate_kernel_ir.
  try {
    CastCpuGeneratedKernel kernel;
    kernel.metadata.kernel_id = generator->next_kernel_id++;
    kernel.metadata.precision = spec->precision;
    kernel.metadata.simd_width = spec->simd_width;
    kernel.metadata.mode = spec->mode;
    kernel.metadata.n_gate_qubits = static_cast<uint32_t>(n_qubits);
    kernel.func_name = "k_" + std::to_string(kernel.metadata.kernel_id);
    kernel.context = std::make_unique<llvm::LLVMContext>();
    kernel.module = std::make_unique<llvm::Module>(kernel.func_name + "_module", *kernel.context);
    if (spec->mode == CAST_CPU_MATRIX_LOAD_STACK_LOAD) {
      kernel.matrix.assign(matrix, matrix + matrix_len);
    }

    auto func = cast_cpu_generate_kernel_ir(*spec, matrix, matrix_len, qubits, n_qubits,
                                            kernel.func_name, *kernel.module);
    if (!func) {
      write_error_message(err_buf, err_buf_len, llvm::toString(func.takeError()));
      return 1;
    }

    generator->kernels.push_back(std::move(kernel));
    *out_kernel_id = generator->kernels.back().metadata.kernel_id;
    clear_error_buffer(err_buf, err_buf_len);
    return 0;
  } catch (const std::exception &ex) {
    write_error_message(err_buf, err_buf_len, ex.what());
    return 1;
  } catch (...) {
    write_error_message(err_buf, err_buf_len, "unknown error in kernel generation");
    return 1;
  }
}

extern "C" int cast_cpu_kernel_generator_request_asm(cast_cpu_kernel_generator_t *generator,
                                                     cast_cpu_kernel_id_t kernel_id, char *err_buf,
                                                     size_t err_buf_len) {
  if (generator == nullptr) {
    write_error_message(err_buf, err_buf_len, "generator must not be null");
    return 1;
  }
  auto it = std::find_if(
      generator->kernels.begin(), generator->kernels.end(),
      [kernel_id](const CastCpuGeneratedKernel &k) { return k.metadata.kernel_id == kernel_id; });
  if (it == generator->kernels.end()) {
    write_error_message(err_buf, err_buf_len, "kernel id not found in generator");
    return 1;
  }
  it->capture_asm = true;
  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

extern "C" int cast_cpu_kernel_generator_emit_ir(cast_cpu_kernel_generator_t *generator,
                                                 cast_cpu_kernel_id_t kernel_id, char *out_ir,
                                                 size_t ir_buf_len, size_t *out_ir_len,
                                                 char *err_buf, size_t err_buf_len) {
  if (generator == nullptr) {
    write_error_message(err_buf, err_buf_len, "generator must not be null");
    return 1;
  }

  auto it = std::find_if(
      generator->kernels.begin(), generator->kernels.end(),
      [kernel_id](const CastCpuGeneratedKernel &k) { return k.metadata.kernel_id == kernel_id; });
  if (it == generator->kernels.end()) {
    write_error_message(err_buf, err_buf_len, "kernel id not found in generator");
    return 1;
  }

  // Optimize (idempotent) and populate it->ir if not done yet.
  if (auto err = cast_cpu_optimize_kernel_ir(*it)) {
    write_error_message(err_buf, err_buf_len, llvm::toString(std::move(err)));
    return 1;
  }

  const std::string &ir = it->ir;
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

extern "C" int cast_cpu_kernel_generator_finish(cast_cpu_kernel_generator_t *generator,
                                                cast_cpu_jit_session_t **out_session, char *err_buf,
                                                size_t err_buf_len) {
  if (generator == nullptr) {
    write_error_message(err_buf, err_buf_len, "generator must not be null");
    return 1;
  }
  if (out_session == nullptr) {
    write_error_message(err_buf, err_buf_len, "out_session must not be null");
    return 1;
  }

  try {
    auto jit = cast_cpu_jit_create(static_cast<unsigned>(std::thread::hardware_concurrency()));
    if (!jit) {
      write_error_message(err_buf, err_buf_len, llvm::toString(jit.takeError()));
      return 1;
    }

    auto *session = new cast_cpu_jit_session_t();
    session->jit = std::move(*jit);
    session->kernels.reserve(generator->kernels.size());
    for (auto &generated : generator->kernels) {
      CastCpuJittedKernel kernel;
      if (auto err = cast_cpu_jit_compile_kernel(*session->jit, generated, kernel)) {
        write_error_message(err_buf, err_buf_len, llvm::toString(std::move(err)));
        delete session;
        return 1;
      }
      const auto kid = kernel.metadata.kernel_id;
      session->kernels.emplace(kid, std::move(kernel));
    }

    *out_session = session;
    clear_error_buffer(err_buf, err_buf_len);

    // ownership transferred; caller must not use generator again
    delete generator;
    return 0;
  } catch (const std::exception &ex) {
    write_error_message(err_buf, err_buf_len, ex.what());
    return 1;
  } catch (...) {
    write_error_message(err_buf, err_buf_len, "unknown error while initializing JIT");
    return 1;
  }
}

extern "C" int cast_cpu_jit_session_apply(cast_cpu_jit_session_t *session,
                                          cast_cpu_kernel_id_t kernel_id, void *sv,
                                          uint32_t n_qubits, cast_cpu_precision_t sv_precision,
                                          cast_cpu_simd_width_t sv_simd_width, int32_t n_threads,
                                          char *err_buf, size_t err_buf_len) {
  if (session == nullptr) {
    write_error_message(err_buf, err_buf_len, "session must not be null");
    return 1;
  }
  const auto *kernel = find_jitted_kernel(session, kernel_id);
  if (kernel == nullptr) {
    write_error_message(err_buf, err_buf_len, "kernel id not found in JIT session");
    return 1;
  }

  if (auto err = cast_cpu_jit_apply_kernel(*kernel, sv, n_qubits, sv_precision, sv_simd_width,
                                           n_threads)) {
    write_error_message(err_buf, err_buf_len, llvm::toString(std::move(err)));
    return 1;
  }
  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

extern "C" int cast_cpu_jit_session_emit_asm(cast_cpu_jit_session_t *session,
                                             cast_cpu_kernel_id_t kernel_id, char *out_asm,
                                             size_t asm_buf_len, size_t *out_asm_len, char *err_buf,
                                             size_t err_buf_len) {
  if (session == nullptr) {
    write_error_message(err_buf, err_buf_len, "session must not be null");
    return 1;
  }
  const auto *kernel = find_jitted_kernel(session, kernel_id);
  if (kernel == nullptr) {
    write_error_message(err_buf, err_buf_len, "kernel id not found in JIT session");
    return 1;
  }

  if (kernel->asm_text.empty()) {
    write_error_message(err_buf, err_buf_len,
                        "assembly was not captured for this kernel; "
                        "call request_asm before init_jit");
    return 1;
  }

  const std::string &asm_text = kernel->asm_text;
  if (out_asm_len != nullptr)
    *out_asm_len = asm_text.size();

  if (out_asm != nullptr && asm_buf_len > 0) {
    const size_t n = std::min(asm_buf_len - 1, asm_text.size());
    std::memcpy(out_asm, asm_text.data(), n);
    out_asm[n] = '\0';
  }

  clear_error_buffer(err_buf, err_buf_len);
  return 0;
}

extern "C" void cast_cpu_jit_session_delete(cast_cpu_jit_session_t *session) { delete session; }
