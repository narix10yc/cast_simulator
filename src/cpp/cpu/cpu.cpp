// Rust ↔ C ABI boundary for the CPU kernel pipeline.
//
// Error handling convention: internal helpers (cast_cpu_generate_kernel_ir,
// cast_cpu_jit_create, cast_cpu_optimize_kernel_ir, cast_cpu_jit_compile_kernel)
// propagate via llvm::Expected<T> / llvm::Error. This file is the *only* place
// those are converted into the C `err_buf` + return-code contract.
//
// The `try { ... } catch (...)` blocks here are the C-ABI safety net: an
// exception must not propagate across extern "C" into Rust (UB). In practice
// only allocation failures (std::bad_alloc from `new`, std::make_unique,
// std::vector growth) reach these handlers — any LLVM/logic error is already
// surfaced via llvm::Error within the protected region. Do NOT add try/catch
// inside helpers that return llvm::Error; keep the boundary thin.

#include "cast_cpu.h"

#include "cpu_gen.h"
#include "cpu_jit.h"
#include "internal/util.h"

#include <llvm/Support/Error.h>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace cast_cpu_detail;

struct cast_cpu_kernel_generator_t {
  std::vector<CastCpuGeneratedKernel> kernels;
  cast_cpu_kernel_id_t next_kernel_id = 0;
};

struct cast_cpu_jit_session_t {
  std::unique_ptr<llvm::orc::LLJIT> jit;
};

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
                                                cast_cpu_jit_session_t **out_session,
                                                cast_cpu_jit_kernel_record_t **out_records,
                                                size_t *out_n_records, char *err_buf,
                                                size_t err_buf_len) {
  if (generator == nullptr) {
    write_error_message(err_buf, err_buf_len, "generator must not be null");
    return 1;
  }
  if (out_session == nullptr || out_records == nullptr || out_n_records == nullptr) {
    write_error_message(err_buf, err_buf_len, "output pointers must not be null");
    return 1;
  }

  try {
    auto jit = cast_cpu_jit_create(std::thread::hardware_concurrency());
    if (!jit) {
      write_error_message(err_buf, err_buf_len, llvm::toString(jit.takeError()));
      return 1;
    }

    const size_t n = generator->kernels.size();
    auto *records = static_cast<cast_cpu_jit_kernel_record_t *>(
        std::calloc(n, sizeof(cast_cpu_jit_kernel_record_t)));
    if (!records && n > 0) {
      write_error_message(err_buf, err_buf_len, "failed to allocate kernel records");
      return 1;
    }

    auto *session = new cast_cpu_jit_session_t();
    session->jit = std::move(*jit);

    for (size_t i = 0; i < n; ++i) {
      if (auto err =
              cast_cpu_jit_compile_kernel(*session->jit, generator->kernels[i], records[i])) {
        // Frees the records' inner fields AND the array itself.
        cast_cpu_jit_kernel_records_free(records, i);
        delete session;
        write_error_message(err_buf, err_buf_len, llvm::toString(std::move(err)));
        return 1;
      }
    }

    *out_session = session;
    *out_records = records;
    *out_n_records = n;
    clear_error_buffer(err_buf, err_buf_len);

    // Ownership transferred; caller must not use generator again.
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

extern "C" void cast_cpu_jit_kernel_records_free(cast_cpu_jit_kernel_record_t *records, size_t n) {
  if (records == nullptr)
    return;
  for (size_t i = 0; i < n; ++i) {
    std::free(records[i].matrix);
    std::free(records[i].asm_text);
  }
  std::free(records);
}

extern "C" void cast_cpu_jit_session_delete(cast_cpu_jit_session_t *session) { delete session; }
