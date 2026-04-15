// Rust ↔ C ABI boundary for the CUDA kernel pipeline.
//
// Error handling convention: internal helpers (cast_cuda_generate_kernel_ir,
// cast_cuda_compile_kernel) propagate via llvm::Expected<T> / llvm::Error.
// This file is the *only* place those are converted into the C `err_buf` +
// return-code contract.
//
// The `try { ... } catch (...)` blocks here are the C-ABI safety net: an
// exception must not propagate across extern "C" into Rust (UB). In practice
// only allocation failures (std::bad_alloc from `new`, std::make_unique,
// std::vector growth) reach these handlers — any LLVM/logic error is already
// surfaced via llvm::Error within the protected region. Do NOT add try/catch
// inside helpers that return llvm::Error; keep the boundary thin.

#include "cast_cuda.h"

#include "cuda_gen.h"
#include "cuda_jit.h"
#include "cuda_util.h"

#include <llvm/Support/Error.h>

#include <cstring>
#include <exception>
#include <memory>
#include <string>

using namespace cast_cuda_detail;

// ── Gate PTX compilation ──────────────────────────────────────────────────────

extern "C" int cast_cuda_compile_gate_ptx(const cast_cuda_kernel_gen_spec_t *spec,
                                          const cast_cuda_complex64_t *matrix, size_t matrix_len,
                                          const uint32_t *qubits, size_t n_qubits, char **out_ptx,
                                          char **out_func_name, uint32_t *out_n_gate_qubits,
                                          uint8_t *out_precision, char *err_buf,
                                          size_t err_buf_len) {
  if (!spec || !matrix || !qubits || !out_ptx || !out_func_name || !out_n_gate_qubits ||
      !out_precision) {
    write_error_message(err_buf, err_buf_len, "arguments must not be null");
    return 1;
  }

  try {
    const std::string func_name = "k_gate";
    auto context = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>(func_name + "_mod", *context);

    auto func = cast_cuda_generate_kernel_ir(*spec, matrix, matrix_len, qubits, n_qubits, func_name,
                                             *module);
    if (!func) {
      write_error_message(err_buf, err_buf_len, llvm::toString(func.takeError()));
      return 1;
    }

    CastCudaGeneratedKernel generated;
    generated.spec = *spec;
    generated.n_gate_qubits = static_cast<uint32_t>(n_qubits);
    generated.func_name = func_name;
    generated.context = std::move(context);
    generated.module = std::move(module);

    CastCudaCompiledKernel compiled;
    if (auto err = cast_cuda_compile_kernel(generated, compiled)) {
      write_error_message(err_buf, err_buf_len, llvm::toString(std::move(err)));
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
    *out_precision = static_cast<uint8_t>(compiled.precision);

    clear_error_buffer(err_buf, err_buf_len);
    return 0;
  } catch (const std::exception &ex) {
    write_error_message(err_buf, err_buf_len, ex.what());
    return 1;
  } catch (...) {
    write_error_message(err_buf, err_buf_len, "unknown error in gate PTX compilation");
    return 1;
  }
}

extern "C" void cast_cuda_str_free(const char *s) { delete[] s; }
