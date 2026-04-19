// Rust ↔ C ABI boundary for the CUDA kernel pipeline.
//
// Error handling convention: internal helpers (cast::cuda::generateKernelIr,
// cast::cuda::compileKernel) propagate via llvm::Expected<T> / llvm::Error.
// This file is the *only* place those are converted into the C `errBuf` +
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

cast::Precision toInternal(cast_precision_t precision) {
  return static_cast<cast::Precision>(precision);
}

cast::cuda::KernelGenSpec toInternal(const cast_cuda_kernel_gen_spec_t &spec) {
  return {
      toInternal(spec.precision), spec.ztol, spec.otol, spec.sm_major, spec.sm_minor, spec.maxnreg};
}

std::vector<cast::Complex64> copyMatrix(const cast_complex64_t *matrix, size_t matrixLen) {
  std::vector<cast::Complex64> out;
  out.reserve(matrixLen);
  for (size_t i = 0; i < matrixLen; ++i) {
    out.push_back({matrix[i].re, matrix[i].im});
  }
  return out;
}

uint8_t toFfi(cast::Precision precision) { return static_cast<uint8_t>(precision); }

bool expectedMatrixLen(size_t nQubits, size_t *outLen) {
  constexpr size_t kBits = sizeof(size_t) * 8;
  if (nQubits >= kBits / 2) {
    return false;
  }
  *outLen = static_cast<size_t>(1) << (2 * nQubits);
  return true;
}

} // namespace

extern "C" int cast_cuda_compile_gate_ptx(const cast_cuda_kernel_gen_spec_t *spec,
                                          const cast_complex64_t *matrix, size_t matrixLen,
                                          const uint32_t *qubits, size_t nQubits, char **out_ptx,
                                          char **out_func_name, uint32_t *out_n_gate_qubits,
                                          uint8_t *out_precision, char *errBuf, size_t errBufLen) {
  if (!spec || !matrix || !qubits || !out_ptx || !out_func_name || !out_n_gate_qubits ||
      !out_precision) {
    writeErrBuf(errBuf, errBufLen, "arguments must not be null");
    return 1;
  }

  try {
    const cast::cuda::KernelGenSpec internalSpec = toInternal(*spec);
    if (!cast::isValidPrecision(internalSpec.precision)) {
      writeErrBuf(errBuf, errBufLen, "spec.precision must be F32 or F64");
      return 1;
    }
    if (qubits == nullptr || nQubits == 0) {
      writeErrBuf(errBuf, errBufLen, "qubits must not be null/empty");
      return 1;
    }
    size_t expectedLen = 0;
    if (!expectedMatrixLen(nQubits, &expectedLen) || expectedLen != matrixLen) {
      writeErrBuf(errBuf, errBufLen, "matrixLen must equal (2^nQubits)^2");
      return 1;
    }
    const std::vector<cast::Complex64> internalMatrix = copyMatrix(matrix, matrixLen);

    const std::string funcName = "k_gate";
    auto context = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>(funcName + "_mod", *context);

    auto func = cast::cuda::generateKernelIr(internalSpec, internalMatrix.data(), matrixLen, qubits,
                                             nQubits, funcName, *module);
    if (!func) {
      writeErrBuf(errBuf, errBufLen, llvm::toString(func.takeError()));
      return 1;
    }

    cast::cuda::GeneratedKernel generated;
    generated.spec = internalSpec;
    generated.nGateQubits = static_cast<uint32_t>(nQubits);
    generated.funcName = funcName;
    generated.context = std::move(context);
    generated.module = std::move(module);

    cast::cuda::CompiledKernel compiled;
    if (auto err = cast::cuda::compileKernel(generated, compiled)) {
      writeErrBuf(errBuf, errBufLen, llvm::toString(std::move(err)));
      return 1;
    }

    // Return PTX (C++-allocated; caller frees with cast_cuda_str_free).
    auto *ptxBuf = new char[compiled.ptx.size() + 1];
    std::memcpy(ptxBuf, compiled.ptx.data(), compiled.ptx.size());
    ptxBuf[compiled.ptx.size()] = '\0';

    // Return funcName (C++-allocated; caller frees with cast_cuda_str_free).
    auto *fnBuf = new char[compiled.funcName.size() + 1];
    std::memcpy(fnBuf, compiled.funcName.data(), compiled.funcName.size());
    fnBuf[compiled.funcName.size()] = '\0';

    *out_ptx = ptxBuf;
    *out_func_name = fnBuf;
    *out_n_gate_qubits = compiled.nGateQubits;
    *out_precision = toFfi(compiled.precision);

    clearErrBuf(errBuf, errBufLen);
    return 0;
  } catch (const std::exception &ex) {
    writeErrBuf(errBuf, errBufLen, ex.what());
    return 1;
  } catch (...) {
    writeErrBuf(errBuf, errBufLen, "unknown error in gate PTX compilation");
    return 1;
  }
}

extern "C" void cast_cuda_str_free(const char *s) { delete[] s; }
