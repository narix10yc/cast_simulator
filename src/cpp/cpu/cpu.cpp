// Rust-C ABI boundary for the CPU kernel pipeline.
//
// Error handling: internal helpers propagate via LLVM error handling system (`llvm::Expected<T>`
// and `llvm::Error).
//
// At the FFI boundary, we use try-catch blocks for possible C++ std::exception (such as
// std::bad_alloc of `new`). These exceptions must not cross the FFI boundary to Rust.
// Rust side needs to catch these flags accordingly (nullptr, return value checks).

#include "../include/ffi_cpu.h"

#include "cpu_gen.hpp"
#include "cpu_jit.hpp"
#include "internal/util.hpp"

#include <llvm/Support/Error.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

static_assert(sizeof(cast_complex64_t) == 16);
static_assert(alignof(cast_complex64_t) == alignof(double));
static_assert(static_cast<int>(CAST_PRECISION_F32) == static_cast<int>(cast::Precision::F32));
static_assert(static_cast<int>(CAST_PRECISION_F64) == static_cast<int>(cast::Precision::F64));
static_assert(static_cast<int>(CAST_CPU_SIMD_WIDTH_W128) ==
              static_cast<int>(cast::cpu::SimdWidth::W128));
static_assert(static_cast<int>(CAST_CPU_SIMD_WIDTH_W256) ==
              static_cast<int>(cast::cpu::SimdWidth::W256));
static_assert(static_cast<int>(CAST_CPU_SIMD_WIDTH_W512) ==
              static_cast<int>(cast::cpu::SimdWidth::W512));
static_assert(static_cast<int>(CAST_CPU_MATRIX_LOAD_IMM_VALUE) ==
              static_cast<int>(cast::cpu::MatrixLoadMode::ImmValue));
static_assert(static_cast<int>(CAST_CPU_MATRIX_LOAD_STACK_LOAD) ==
              static_cast<int>(cast::cpu::MatrixLoadMode::StackLoad));

cast::Precision to_internal(cast_precision_t precision) {
  return static_cast<cast::Precision>(precision);
}

cast::cpu::SimdWidth to_internal(cast_cpu_simd_width_t width) {
  return static_cast<cast::cpu::SimdWidth>(width);
}

cast::cpu::MatrixLoadMode to_internal(cast_cpu_matrix_load_mode_t mode) {
  return static_cast<cast::cpu::MatrixLoadMode>(mode);
}

cast::cpu::KernelGenSpec to_internal_spec(const cast_cpu_kernel_gen_request_t &request) {
  return {to_internal(request.precision), to_internal(request.simd_width),
          to_internal(request.mode), request.ztol, request.otol};
}

std::vector<cast::Complex64> copy_matrix(const cast_complex64_t *matrix, size_t matrix_len) {
  std::vector<cast::Complex64> out;
  out.reserve(matrix_len);
  for (size_t i = 0; i < matrix_len; ++i) {
    out.push_back({matrix[i].re, matrix[i].im});
  }
  return out;
}

cast_precision_t to_ffi(cast::Precision precision) {
  return static_cast<cast_precision_t>(precision);
}

cast_cpu_simd_width_t to_ffi(cast::cpu::SimdWidth width) {
  return static_cast<cast_cpu_simd_width_t>(width);
}

cast_cpu_matrix_load_mode_t to_ffi(cast::cpu::MatrixLoadMode mode) {
  return static_cast<cast_cpu_matrix_load_mode_t>(mode);
}

cast_cpu_kernel_metadata_t to_ffi(const cast::cpu::KernelMetadata &metadata) {
  return {metadata.kernel_id, to_ffi(metadata.precision), to_ffi(metadata.simd_width),
          to_ffi(metadata.mode), metadata.n_gate_qubits};
}

llvm::Error copy_to_ffi(const cast::cpu::CompiledKernelRecord &src,
                        cast_cpu_jit_kernel_record_t &dst) {
  dst = {};
  dst.metadata = to_ffi(src.metadata);
  dst.entry = src.entry;
  dst.matrix_len = src.matrix.size();

  if (!src.matrix.empty()) {
    const size_t nbytes = src.matrix.size() * sizeof(cast_complex64_t);
    dst.matrix = static_cast<cast_complex64_t *>(std::malloc(nbytes));
    if (!dst.matrix)
      return llvm::createStringError("failed to allocate matrix buffer");
    for (size_t i = 0; i < src.matrix.size(); ++i) {
      dst.matrix[i] = {src.matrix[i].re, src.matrix[i].im};
    }
  }

  if (src.asm_text.has_value()) {
    dst.asm_text = static_cast<char *>(std::malloc(src.asm_text->size() + 1));
    if (!dst.asm_text) {
      std::free(dst.matrix);
      dst.matrix = nullptr;
      dst.matrix_len = 0;
      return llvm::createStringError("failed to allocate asm text buffer");
    }
    std::memcpy(dst.asm_text, src.asm_text->data(), src.asm_text->size());
    dst.asm_text[src.asm_text->size()] = '\0';
  }

  return llvm::Error::success();
}

} // namespace

struct cast_cpu_kernel_generator_t {
  std::vector<cast::cpu::GeneratedKernel> kernels;
  // Kernel ids start at 1; 0 is reserved as the FFI failure sentinel for
  // cast_cpu_kernel_generator_generate.
  cast_cpu_kernel_id_t next_kernel_id = 1;
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

extern "C" cast_cpu_kernel_id_t cast_cpu_kernel_generator_generate(
    cast_cpu_kernel_generator_t *generator, const cast_cpu_kernel_gen_request_t *request,
    char *err_buf, size_t err_buf_len) {
  if (generator == nullptr) {
    write_err_buf(err_buf, err_buf_len, "generator must not be null");
    return 0;
  }
  if (request == nullptr) {
    write_err_buf(err_buf, err_buf_len, "request must not be null");
    return 0;
  }
  // Dereferencing a null matrix pointer in copy_matrix is UB; guard here.
  // Shape-level validation (length, qubits, spec fields) is delegated to
  // generate_kernel_ir.
  if (request->matrix == nullptr && request->matrix_len != 0) {
    write_err_buf(err_buf, err_buf_len, "request->matrix must not be null");
    return 0;
  }

  try {
    const cast::cpu::KernelGenSpec internal_spec = to_internal_spec(*request);
    std::vector<cast::Complex64> matrix_buf =
        copy_matrix(request->matrix, request->matrix_len);

    cast::cpu::GeneratedKernel kernel;
    kernel.metadata.kernel_id = generator->next_kernel_id++;
    kernel.metadata.precision = internal_spec.precision;
    kernel.metadata.simd_width = internal_spec.simd_width;
    kernel.metadata.mode = internal_spec.mode;
    kernel.metadata.n_gate_qubits = static_cast<uint32_t>(request->n_qubits);
    kernel.func_name = "k_" + std::to_string(kernel.metadata.kernel_id);
    kernel.context = std::make_unique<llvm::LLVMContext>();
    kernel.module = std::make_unique<llvm::Module>(kernel.func_name + "_module", *kernel.context);
    kernel.capture_asm = request->capture_asm;

    auto func = cast::cpu::generate_kernel_ir(internal_spec, matrix_buf.data(),
                                              request->matrix_len, request->qubits,
                                              request->n_qubits, kernel.func_name, *kernel.module);
    if (!func) {
      write_err_buf(err_buf, err_buf_len, llvm::toString(func.takeError()));
      return 0;
    }

    // StackLoad mode needs the live matrix values at apply time.  ImmValue
    // mode bakes them into the IR, so the buffer can be dropped.
    if (internal_spec.mode == cast::cpu::MatrixLoadMode::StackLoad) {
      kernel.matrix = std::move(matrix_buf);
    }

    if (request->capture_ir) {
      if (auto err = cast::cpu::optimize_kernel_ir(kernel)) {
        write_err_buf(err_buf, err_buf_len, llvm::toString(std::move(err)));
        return 0;
      }
    }

    const cast_cpu_kernel_id_t kernel_id = kernel.metadata.kernel_id;
    generator->kernels.push_back(std::move(kernel));
    clear_err_buf(err_buf, err_buf_len);
    return kernel_id;
  } catch (const std::exception &ex) {
    write_err_buf(err_buf, err_buf_len, ex.what());
    return 0;
  } catch (...) {
    write_err_buf(err_buf, err_buf_len, "unknown error in kernel generation");
    return 0;
  }
}

extern "C" int cast_cpu_kernel_generator_emit_ir(cast_cpu_kernel_generator_t *generator,
                                                 cast_cpu_kernel_id_t kernel_id, char *out_ir,
                                                 size_t ir_buf_len, size_t *out_ir_len,
                                                 char *err_buf, size_t err_buf_len) {
  if (generator == nullptr) {
    write_err_buf(err_buf, err_buf_len, "generator must not be null");
    return 1;
  }

  auto it = std::find_if(generator->kernels.begin(), generator->kernels.end(),
                         [kernel_id](const cast::cpu::GeneratedKernel &k) {
                           return k.metadata.kernel_id == kernel_id;
                         });
  if (it == generator->kernels.end()) {
    write_err_buf(err_buf, err_buf_len, "kernel id not found in generator");
    return 1;
  }

  // Optimize (idempotent) and populate it->ir if not done yet.
  if (auto err = cast::cpu::optimize_kernel_ir(*it)) {
    write_err_buf(err_buf, err_buf_len, llvm::toString(std::move(err)));
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

  clear_err_buf(err_buf, err_buf_len);
  return 0;
}

extern "C" int cast_cpu_kernel_generator_finish(cast_cpu_kernel_generator_t *generator,
                                                cast_cpu_jit_session_t **out_session,
                                                cast_cpu_jit_kernel_record_t **out_records,
                                                size_t *out_n_records, char *err_buf,
                                                size_t err_buf_len) {
  if (generator == nullptr) {
    write_err_buf(err_buf, err_buf_len, "generator must not be null");
    return 1;
  }
  if (out_session == nullptr || out_records == nullptr || out_n_records == nullptr) {
    write_err_buf(err_buf, err_buf_len, "output pointers must not be null");
    return 1;
  }

  try {
    auto jit = cast::cpu::jit_create(std::thread::hardware_concurrency());
    if (!jit) {
      write_err_buf(err_buf, err_buf_len, llvm::toString(jit.takeError()));
      return 1;
    }

    const size_t n = generator->kernels.size();
    auto *records = static_cast<cast_cpu_jit_kernel_record_t *>(
        std::calloc(n, sizeof(cast_cpu_jit_kernel_record_t)));
    if (!records && n > 0) {
      write_err_buf(err_buf, err_buf_len, "failed to allocate kernel records");
      return 1;
    }

    auto *session = new cast_cpu_jit_session_t();
    session->jit = std::move(*jit);

    for (size_t i = 0; i < n; ++i) {
      auto compiled = cast::cpu::jit_compile_kernel(*session->jit, generator->kernels[i]);
      if (!compiled) {
        // Frees the records' inner fields AND the array itself.
        cast_cpu_jit_kernel_records_free(records, i);
        delete session;
        write_err_buf(err_buf, err_buf_len, llvm::toString(compiled.takeError()));
        return 1;
      }
      if (auto err = copy_to_ffi(*compiled, records[i])) {
        cast_cpu_jit_kernel_records_free(records, i + 1);
        delete session;
        write_err_buf(err_buf, err_buf_len, llvm::toString(std::move(err)));
        return 1;
      }
    }

    *out_session = session;
    *out_records = records;
    *out_n_records = n;
    clear_err_buf(err_buf, err_buf_len);

    // Ownership transferred; caller must not use generator again.
    delete generator;
    return 0;
  } catch (const std::exception &ex) {
    write_err_buf(err_buf, err_buf_len, ex.what());
    return 1;
  } catch (...) {
    write_err_buf(err_buf, err_buf_len, "unknown error while initializing JIT");
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
