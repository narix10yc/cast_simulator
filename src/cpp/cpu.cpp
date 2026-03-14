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
#include <utility>
#include <vector>

namespace {

bool is_valid_precision(cast_cpu_precision_t precision) {
  return precision == CAST_CPU_PRECISION_F32 ||
         precision == CAST_CPU_PRECISION_F64;
}

bool is_valid_simd_width(cast_cpu_simd_width_t simd_width) {
  return simd_width == CAST_CPU_SIMD_WIDTH_W128 ||
         simd_width == CAST_CPU_SIMD_WIDTH_W256 ||
         simd_width == CAST_CPU_SIMD_WIDTH_W512;
}

bool is_valid_mode(cast_cpu_matrix_load_mode_t mode) {
  return mode == CAST_CPU_MATRIX_LOAD_IMM_VALUE ||
         mode == CAST_CPU_MATRIX_LOAD_STACK_LOAD;
}

void write_error(char* err_buf, size_t err_buf_len, const std::string& msg) {
  if (err_buf == nullptr || err_buf_len == 0) {
    return;
  }
  const size_t n = std::min(err_buf_len - 1, msg.size());
  std::memcpy(err_buf, msg.data(), n);
  err_buf[n] = '\0';
}

void clear_error(char* err_buf, size_t err_buf_len) {
  if (err_buf != nullptr && err_buf_len > 0) {
    err_buf[0] = '\0';
  }
}

bool expected_matrix_len(size_t n_qubits, size_t* out_len) {
  constexpr size_t kBits = sizeof(size_t) * 8;
  if (n_qubits > (kBits / 2)) {
    return false;
  }
  const size_t exponent = n_qubits * 2;
  *out_len = static_cast<size_t>(1) << exponent;
  return true;
}

int validate_generate_args(const cast_cpu_kernel_generator_t* generator,
                           const cast_cpu_kernel_gen_spec_t* spec,
                           const cast_cpu_complex64_t* matrix,
                           size_t matrix_len,
                           const uint32_t* qubits,
                           size_t n_qubits,
                           cast_cpu_kernel_id_t* out_kernel_id,
                           char* err_buf,
                           size_t err_buf_len) {
  if (generator == nullptr) {
    write_error(err_buf, err_buf_len, "generator must not be null");
    return 1;
  }
  if (spec == nullptr) {
    write_error(err_buf, err_buf_len, "spec must not be null");
    return 1;
  }
  if (matrix == nullptr) {
    write_error(err_buf, err_buf_len, "matrix must not be null");
    return 1;
  }
  if (qubits == nullptr && n_qubits != 0) {
    write_error(err_buf, err_buf_len, "qubits must not be null");
    return 1;
  }
  if (out_kernel_id == nullptr) {
    write_error(err_buf, err_buf_len, "out_kernel_id must not be null");
    return 1;
  }
  if (!is_valid_precision(spec->precision)) {
    write_error(err_buf, err_buf_len, "invalid precision");
    return 1;
  }
  if (!is_valid_simd_width(spec->simd_width)) {
    write_error(err_buf, err_buf_len, "invalid SIMD width");
    return 1;
  }
  if (!is_valid_mode(spec->mode)) {
    write_error(err_buf, err_buf_len, "invalid matrix load mode");
    return 1;
  }
  if (spec->ztol < 0.0 || spec->otol < 0.0) {
    write_error(err_buf, err_buf_len, "tolerances must be non-negative");
    return 1;
  }

  size_t expected_len = 0;
  if (!expected_matrix_len(n_qubits, &expected_len)) {
    write_error(err_buf, err_buf_len, "too many qubits for dense matrix input");
    return 1;
  }
  if (matrix_len != expected_len) {
    write_error(err_buf,
                err_buf_len,
                "matrix length does not match the target qubit count");
    return 1;
  }

  for (size_t i = 1; i < n_qubits; ++i) {
    if (qubits[i - 1] >= qubits[i]) {
      write_error(err_buf, err_buf_len, "qubits must be strictly ascending");
      return 1;
    }
  }

  clear_error(err_buf, err_buf_len);
  return 0;
}

} // namespace

struct cast_cpu_kernel_generator_t {
  std::vector<CastCpuGeneratedKernel> kernels{};
  cast_cpu_kernel_id_t next_kernel_id = 0;
};

struct cast_cpu_jit_session_t {
  std::unique_ptr<llvm::orc::LLJIT> jit{};
  std::vector<CastCpuJittedKernel> kernels{};
};

namespace {

const CastCpuJittedKernel* find_kernel(const cast_cpu_jit_session_t* session,
                                       cast_cpu_kernel_id_t kernel_id) {
  if (session == nullptr) {
    return nullptr;
  }
  for (const auto& kernel : session->kernels) {
    if (kernel.metadata.kernel_id == kernel_id) {
      return &kernel;
    }
  }
  return nullptr;
}

} // namespace

extern "C" cast_cpu_kernel_generator_t* cast_cpu_kernel_generator_new(void) {
  try {
    return new cast_cpu_kernel_generator_t();
  } catch (...) {
    return nullptr;
  }
}

extern "C" void
cast_cpu_kernel_generator_delete(cast_cpu_kernel_generator_t* generator) {
  delete generator;
}

extern "C" int
cast_cpu_kernel_generator_generate(cast_cpu_kernel_generator_t* generator,
                                   const cast_cpu_kernel_gen_spec_t* spec,
                                   const cast_cpu_complex64_t* matrix,
                                   size_t matrix_len,
                                   const uint32_t* qubits,
                                   size_t n_qubits,
                                   cast_cpu_kernel_id_t* out_kernel_id,
                                   char* err_buf,
                                   size_t err_buf_len) {
  if (validate_generate_args(generator,
                             spec,
                             matrix,
                             matrix_len,
                             qubits,
                             n_qubits,
                             out_kernel_id,
                             err_buf,
                             err_buf_len) != 0) {
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
    kernel.module = std::make_unique<llvm::Module>(kernel.func_name + "_module",
                                                   *kernel.context);
    if (spec->mode == CAST_CPU_MATRIX_LOAD_STACK_LOAD) {
      kernel.matrix.assign(matrix, matrix + matrix_len);
    }

    auto func = cast_cpu_generate_kernel_ir(*spec,
                                            matrix,
                                            matrix_len,
                                            qubits,
                                            n_qubits,
                                            kernel.func_name,
                                            *kernel.module);
    if (!func) {
      write_error(err_buf, err_buf_len, llvm::toString(func.takeError()));
      return 1;
    }

    generator->kernels.push_back(std::move(kernel));
    *out_kernel_id = generator->kernels.back().metadata.kernel_id;
    clear_error(err_buf, err_buf_len);
    return 0;
  } catch (const std::exception& ex) {
    write_error(err_buf, err_buf_len, ex.what());
    return 1;
  } catch (...) {
    write_error(err_buf, err_buf_len, "unknown error in kernel generation");
    return 1;
  }
}

extern "C" int
cast_cpu_kernel_generator_finish(cast_cpu_kernel_generator_t* generator,
                                 cast_cpu_jit_session_t** out_session,
                                 char* err_buf,
                                 size_t err_buf_len) {
  if (generator == nullptr) {
    write_error(err_buf, err_buf_len, "generator must not be null");
    return 1;
  }
  if (out_session == nullptr) {
    write_error(err_buf, err_buf_len, "out_session must not be null");
    return 1;
  }

  try {
    auto jit = cast_cpu_jit_create(
        static_cast<unsigned>(std::thread::hardware_concurrency()));
    if (!jit) {
      write_error(err_buf, err_buf_len, llvm::toString(jit.takeError()));
      return 1;
    }

    auto* session = new cast_cpu_jit_session_t();
    session->jit = std::move(*jit);
    session->kernels.reserve(generator->kernels.size());
    for (auto& generated : generator->kernels) {
      CastCpuJittedKernel kernel;
      if (auto err =
              cast_cpu_jit_compile_kernel(*session->jit, generated, kernel)) {
        write_error(err_buf, err_buf_len, llvm::toString(std::move(err)));
        delete session;
        return 1;
      }
      session->kernels.push_back(std::move(kernel));
    }

    *out_session = session;
    clear_error(err_buf, err_buf_len);
    delete generator;
    return 0;
  } catch (const std::exception& ex) {
    write_error(err_buf, err_buf_len, ex.what());
    return 1;
  } catch (...) {
    write_error(err_buf, err_buf_len, "unknown error while initializing JIT");
    return 1;
  }
}

extern "C" int cast_cpu_jit_session_apply(cast_cpu_jit_session_t* session,
                                          cast_cpu_kernel_id_t kernel_id,
                                          void* sv,
                                          uint32_t n_qubits,
                                          cast_cpu_precision_t sv_precision,
                                          cast_cpu_simd_width_t sv_simd_width,
                                          int32_t n_threads,
                                          char* err_buf,
                                          size_t err_buf_len) {
  if (session == nullptr) {
    write_error(err_buf, err_buf_len, "session must not be null");
    return 1;
  }
  const auto* kernel = find_kernel(session, kernel_id);
  if (kernel == nullptr) {
    write_error(err_buf, err_buf_len, "kernel id not found in JIT session");
    return 1;
  }

  if (auto err = cast_cpu_jit_apply_kernel(
          *kernel, sv, n_qubits, sv_precision, sv_simd_width, n_threads)) {
    write_error(err_buf, err_buf_len, llvm::toString(std::move(err)));
    return 1;
  }
  clear_error(err_buf, err_buf_len);
  return 0;
}

extern "C" void cast_cpu_jit_session_delete(cast_cpu_jit_session_t* session) {
  delete session;
}
