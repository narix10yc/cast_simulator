// Rust ↔ C ABI boundary for the CUDA kernel pipeline.
//
// Error handling convention: internal helpers (cast::cuda::generate_kernel_ir,
// cast::cuda::compile_kernel) propagate via llvm::Expected<T> / llvm::Error.
// This file is the *only* place those are converted into the C `err_buf` +
// return-code contract.
//
// The `try { ... } catch (...)` blocks here are the C-ABI safety net: an
// exception must not propagate across extern "C" into Rust (UB). In practice
// only allocation failures (std::bad_alloc from `new`, std::make_unique,
// std::vector growth) reach these handlers — any LLVM/logic error is already
// surfaced via llvm::Error within the protected region. Do NOT add try/catch
// inside helpers that return llvm::Error; keep the boundary thin.

#include "../include/ffi_cuda.h"

#include "cuda_gen.hpp"
#include "cuda_jit.hpp"
#include "cuda_util.hpp"

#include <llvm/Support/Error.h>

#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

// ── Gate PTX compilation ──────────────────────────────────────────────────────

namespace {

static_assert(sizeof(cast_complex64_t) == 16);
static_assert(alignof(cast_complex64_t) == alignof(double));
static_assert(static_cast<int>(CAST_PRECISION_F32) == static_cast<int>(cast::Precision::F32));
static_assert(static_cast<int>(CAST_PRECISION_F64) == static_cast<int>(cast::Precision::F64));

cast::Precision to_internal(cast_precision_t precision) {
  return static_cast<cast::Precision>(precision);
}

cast::cuda::KernelGenSpec to_internal(const cast_cuda_kernel_gen_spec_t &spec) {
  return {to_internal(spec.precision),
          spec.ztol,
          spec.otol,
          spec.sm_major,
          spec.sm_minor,
          spec.maxnreg};
}

std::vector<cast::Complex64> copy_matrix(const cast_complex64_t *matrix, size_t matrix_len) {
  std::vector<cast::Complex64> out;
  out.reserve(matrix_len);
  for (size_t i = 0; i < matrix_len; ++i) {
    out.push_back({matrix[i].re, matrix[i].im});
  }
  return out;
}

uint8_t to_ffi(cast::Precision precision) { return static_cast<uint8_t>(precision); }

bool expected_matrix_len(size_t n_qubits, size_t *out_len) {
  constexpr size_t kBits = sizeof(size_t) * 8;
  if (n_qubits >= kBits / 2) {
    return false;
  }
  *out_len = static_cast<size_t>(1) << (2 * n_qubits);
  return true;
}

} // namespace

extern "C" int cast_cuda_compile_gate_ptx(const cast_cuda_kernel_gen_spec_t *spec,
                                          const cast_complex64_t *matrix, size_t matrix_len,
                                          const uint32_t *qubits, size_t n_qubits, char **out_ptx,
                                          char **out_func_name, uint32_t *out_n_gate_qubits,
                                          uint8_t *out_precision, char *err_buf,
                                          size_t err_buf_len) {
  if (!spec || !matrix || !qubits || !out_ptx || !out_func_name || !out_n_gate_qubits ||
      !out_precision) {
    write_err_buf(err_buf, err_buf_len, "arguments must not be null");
    return 1;
  }

  try {
    const cast::cuda::KernelGenSpec internal_spec = to_internal(*spec);
    if (!cast::is_valid_precision(internal_spec.precision)) {
      write_err_buf(err_buf, err_buf_len, "spec.precision must be F32 or F64");
      return 1;
    }
    if (qubits == nullptr || n_qubits == 0) {
      write_err_buf(err_buf, err_buf_len, "qubits must not be null/empty");
      return 1;
    }
    size_t expected_len = 0;
    if (!expected_matrix_len(n_qubits, &expected_len) || expected_len != matrix_len) {
      write_err_buf(err_buf, err_buf_len, "matrix_len must equal (2^n_qubits)^2");
      return 1;
    }
    const std::vector<cast::Complex64> internal_matrix = copy_matrix(matrix, matrix_len);

    const std::string func_name = "k_gate";
    auto context = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>(func_name + "_mod", *context);

    auto func = cast::cuda::generate_kernel_ir(internal_spec, internal_matrix.data(), matrix_len,
                                               qubits, n_qubits, func_name, *module);
    if (!func) {
      write_err_buf(err_buf, err_buf_len, llvm::toString(func.takeError()));
      return 1;
    }

    cast::cuda::GeneratedKernel generated;
    generated.spec = internal_spec;
    generated.n_gate_qubits = static_cast<uint32_t>(n_qubits);
    generated.func_name = func_name;
    generated.context = std::move(context);
    generated.module = std::move(module);

    cast::cuda::CompiledKernel compiled;
    if (auto err = cast::cuda::compile_kernel(generated, compiled)) {
      write_err_buf(err_buf, err_buf_len, llvm::toString(std::move(err)));
      return 1;
    }

    // Return PTX (C++-allocated; caller frees with cast_cuda_str_free).
    auto *ptx_buf = new char[compiled.ptx.size() + 1];
    std::memcpy(ptx_buf, compiled.ptx.data(), compiled.ptx.size());
    ptx_buf[compiled.ptx.size()] = '\0';

    // Return func_name (C++-allocated; caller frees with cast_cuda_str_free).
    auto *fn_buf = new char[compiled.func_name.size() + 1];
    std::memcpy(fn_buf, compiled.func_name.data(), compiled.func_name.size());
    fn_buf[compiled.func_name.size()] = '\0';

    *out_ptx = ptx_buf;
    *out_func_name = fn_buf;
    *out_n_gate_qubits = compiled.n_gate_qubits;
    *out_precision = to_ffi(compiled.precision);

    clear_err_buf(err_buf, err_buf_len);
    return 0;
  } catch (const std::exception &ex) {
    write_err_buf(err_buf, err_buf_len, ex.what());
    return 1;
  } catch (...) {
    write_err_buf(err_buf, err_buf_len, "unknown error in gate PTX compilation");
    return 1;
  }
}

extern "C" void cast_cuda_str_free(const char *s) { delete[] s; }
