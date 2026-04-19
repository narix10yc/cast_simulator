#include "../include/ffi_cpu.h"

#include "cpu_gen.hpp"
#include "cpu_jit.hpp"
#include "internal/util.hpp"

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// FFI-internal type conversion helpers
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

cast::Precision toInternal(cast_precision_t precision) {
  return static_cast<cast::Precision>(precision);
}

cast::cpu::SimdWidth toInternal(cast_cpu_simd_width_t width) {
  return static_cast<cast::cpu::SimdWidth>(width);
}

cast::cpu::MatrixLoadMode toInternal(cast_cpu_matrix_load_mode_t mode) {
  return static_cast<cast::cpu::MatrixLoadMode>(mode);
}

cast::cpu::KernelGenSpec toInternalSpec(const cast_cpu_kernel_gen_request_t &request) {
  return {toInternal(request.precision), toInternal(request.simd_width), toInternal(request.mode),
          request.ztol, request.otol};
}

std::vector<cast::Complex64> copyMatrix(const cast_complex64_t *matrix, size_t matrixLen) {
  std::vector<cast::Complex64> out;
  out.reserve(matrixLen);
  for (size_t i = 0; i < matrixLen; ++i) {
    out.push_back({matrix[i].re, matrix[i].im});
  }
  return out;
}

cast_precision_t toFfi(cast::Precision precision) {
  return static_cast<cast_precision_t>(precision);
}

cast_cpu_simd_width_t toFfi(cast::cpu::SimdWidth width) {
  return static_cast<cast_cpu_simd_width_t>(width);
}

cast_cpu_matrix_load_mode_t toFfi(cast::cpu::MatrixLoadMode mode) {
  return static_cast<cast_cpu_matrix_load_mode_t>(mode);
}

cast_cpu_kernel_metadata_t toFfi(const cast::cpu::KernelMetadata &metadata) {
  return {metadata.kernelId, toFfi(metadata.precision), toFfi(metadata.simdWidth),
          toFfi(metadata.mode), metadata.nGateQubits};
}

llvm::Error copyToFfi(const cast::cpu::CompiledKernelRecord &src,
                      cast_cpu_jit_kernel_record_t &dst) {
  dst.metadata = toFfi(src.metadata);
  dst.entry = src.entry;
  dst.matrix = nullptr;
  dst.matrix_len = src.matrix.size();
  dst.ir_text = nullptr;
  dst.asm_text = nullptr;

  if (!src.matrix.empty()) {
    const size_t nbytes = src.matrix.size() * sizeof(cast_complex64_t);
    dst.matrix = static_cast<cast_complex64_t *>(std::malloc(nbytes));
    if (!dst.matrix)
      return llvm::createStringError("failed to allocate matrix buffer");
    for (size_t i = 0; i < src.matrix.size(); ++i) {
      dst.matrix[i] = {src.matrix[i].re, src.matrix[i].im};
    }
  }

  auto dupToCString = [](const std::string &s) -> char * {
    char *buf = static_cast<char *>(std::malloc(s.size() + 1));
    if (!buf)
      return nullptr;
    std::memcpy(buf, s.data(), s.size());
    buf[s.size()] = '\0';
    return buf;
  };

  if (src.irText.has_value()) {
    dst.ir_text = dupToCString(*src.irText);
    if (!dst.ir_text) {
      std::free(dst.matrix);
      dst.matrix = nullptr;
      dst.matrix_len = 0;
      return llvm::createStringError("failed to allocate ir text buffer");
    }
  }

  if (src.asmText.has_value()) {
    dst.asm_text = dupToCString(*src.asmText);
    if (!dst.asm_text) {
      std::free(dst.matrix);
      dst.matrix = nullptr;
      dst.matrix_len = 0;
      std::free(dst.ir_text);
      dst.ir_text = nullptr;
      return llvm::createStringError("failed to allocate asm text buffer");
    }
  }

  return llvm::Error::success();
}

} // namespace

struct cast_cpu_kernel_generator_t {
  std::vector<cast::cpu::GeneratedKernel> kernels;
  // Kernel ids start at 1; 0 is reserved as the FFI failure sentinel for
  // cast_cpu_kernel_generator_generate.
  cast_cpu_kernel_id_t nextKernelId = 1;
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

extern "C" cast_cpu_kernel_id_t
cast_cpu_kernel_generator_generate(cast_cpu_kernel_generator_t *generator,
                                   const cast_cpu_kernel_gen_request_t *request, char *errBuf,
                                   size_t errBufLen) {
  if (generator == nullptr) {
    writeErrBuf(errBuf, errBufLen, "generator must not be null");
    return 0;
  }
  if (request == nullptr) {
    writeErrBuf(errBuf, errBufLen, "request must not be null");
    return 0;
  }
  // Dereferencing a null matrix pointer in copyMatrix is UB; guard here.
  // Shape-level validation (length, qubits, spec fields) is delegated to
  // generateKernelIr.
  if (request->matrix == nullptr && request->matrix_len != 0) {
    writeErrBuf(errBuf, errBufLen, "request->matrix must not be null");
    return 0;
  }

  try {
    const cast::cpu::KernelGenSpec internalSpec = toInternalSpec(*request);
    std::vector<cast::Complex64> matrixBuf = copyMatrix(request->matrix, request->matrix_len);

    cast::cpu::GeneratedKernel kernel;
    kernel.metadata.kernelId = generator->nextKernelId++;
    kernel.metadata.precision = internalSpec.precision;
    kernel.metadata.simdWidth = internalSpec.simdWidth;
    kernel.metadata.mode = internalSpec.mode;
    kernel.metadata.nGateQubits = static_cast<uint32_t>(request->n_qubits);
    kernel.funcName = "k_" + std::to_string(kernel.metadata.kernelId);
    kernel.context = std::make_unique<llvm::LLVMContext>();
    kernel.module = std::make_unique<llvm::Module>(kernel.funcName + "_module", *kernel.context);
    kernel.captureIr = request->capture_ir;
    kernel.captureAsm = request->capture_asm;

    auto func = cast::cpu::generateKernelIr(internalSpec, matrixBuf.data(), request->matrix_len,
                                            request->qubits, request->n_qubits, kernel.funcName,
                                            *kernel.module);
    if (!func) {
      writeErrBuf(errBuf, errBufLen, llvm::toString(func.takeError()));
      return 0;
    }

    // StackLoad mode needs the live matrix values at apply time.  ImmValue
    // mode bakes them into the IR, so the buffer can be dropped.
    if (internalSpec.mode == cast::cpu::MatrixLoadMode::StackLoad) {
      kernel.matrix = std::move(matrixBuf);
    }

    const cast_cpu_kernel_id_t kernelId = kernel.metadata.kernelId;
    generator->kernels.push_back(std::move(kernel));
    clearErrBuf(errBuf, errBufLen);
    return kernelId;
  } catch (const std::exception &ex) {
    writeErrBuf(errBuf, errBufLen, ex.what());
    return 0;
  } catch (...) {
    writeErrBuf(errBuf, errBufLen, "unknown error in kernel generation");
    return 0;
  }
}

extern "C" int cast_cpu_kernel_generator_finish(cast_cpu_kernel_generator_t *generator,
                                                cast_cpu_jit_session_t **out_session,
                                                cast_cpu_jit_kernel_record_t **out_records,
                                                size_t *out_n_records, char *errBuf,
                                                size_t errBufLen) {
  if (generator == nullptr) {
    writeErrBuf(errBuf, errBufLen, "generator must not be null");
    return 1;
  }
  if (out_session == nullptr || out_records == nullptr || out_n_records == nullptr) {
    writeErrBuf(errBuf, errBufLen, "output pointers must not be null");
    return 1;
  }

  try {
    auto jit = cast::cpu::createJit(std::thread::hardware_concurrency());
    if (!jit) {
      writeErrBuf(errBuf, errBufLen, llvm::toString(jit.takeError()));
      return 1;
    }

    const size_t n = generator->kernels.size();
    auto *records = static_cast<cast_cpu_jit_kernel_record_t *>(
        std::calloc(n, sizeof(cast_cpu_jit_kernel_record_t)));
    if (!records && n > 0) {
      writeErrBuf(errBuf, errBufLen, "failed to allocate kernel records");
      return 1;
    }

    auto *session = new cast_cpu_jit_session_t();
    session->jit = std::move(*jit);

    for (size_t i = 0; i < n; ++i) {
      auto compiled = cast::cpu::jitCompileKernel(*session->jit, generator->kernels[i]);
      if (!compiled) {
        // Frees the records' inner fields AND the array itself.
        cast_cpu_jit_kernel_records_free(records, i);
        delete session;
        writeErrBuf(errBuf, errBufLen, llvm::toString(compiled.takeError()));
        return 1;
      }
      if (auto err = copyToFfi(*compiled, records[i])) {
        cast_cpu_jit_kernel_records_free(records, i + 1);
        delete session;
        writeErrBuf(errBuf, errBufLen, llvm::toString(std::move(err)));
        return 1;
      }
    }

    *out_session = session;
    *out_records = records;
    *out_n_records = n;
    clearErrBuf(errBuf, errBufLen);

    // Ownership transferred; caller must not use generator again.
    delete generator;
    return 0;
  } catch (const std::exception &ex) {
    writeErrBuf(errBuf, errBufLen, ex.what());
    return 1;
  } catch (...) {
    writeErrBuf(errBuf, errBufLen, "unknown error while initializing JIT");
    return 1;
  }
}

extern "C" void cast_cpu_jit_kernel_records_free(cast_cpu_jit_kernel_record_t *records, size_t n) {
  if (records == nullptr)
    return;
  for (size_t i = 0; i < n; ++i) {
    std::free(records[i].matrix);
    std::free(records[i].ir_text);
    std::free(records[i].asm_text);
  }
  std::free(records);
}

extern "C" void cast_cpu_jit_session_delete(cast_cpu_jit_session_t *session) { delete session; }
