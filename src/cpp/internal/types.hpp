#ifndef CAST_SIMULATOR_SRC_CPP_INTERNAL_TYPES_HPP
#define CAST_SIMULATOR_SRC_CPP_INTERNAL_TYPES_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cast {

enum class Precision : uint8_t {
  F32 = 0,
  F64 = 1,
};

struct Complex64 {
  double re = 0.0;
  double im = 0.0;
};

inline bool is_valid_precision(Precision p) {
  return p == Precision::F32 || p == Precision::F64;
}

namespace cpu {

using KernelId = uint64_t;
using KernelEntry = void(void *);

enum class SimdWidth : uint16_t {
  W128 = 128,
  W256 = 256,
  W512 = 512,
};

enum class MatrixLoadMode : uint8_t {
  ImmValue = 0,
  StackLoad = 1,
};

struct KernelGenSpec {
  Precision precision = Precision::F64;
  SimdWidth simd_width = SimdWidth::W256;
  MatrixLoadMode mode = MatrixLoadMode::ImmValue;
  double ztol = 0.0;
  double otol = 0.0;
};

struct KernelMetadata {
  KernelId kernel_id = 0;
  Precision precision = Precision::F64;
  SimdWidth simd_width = SimdWidth::W256;
  MatrixLoadMode mode = MatrixLoadMode::ImmValue;
  uint32_t n_gate_qubits = 0;
};

struct CompiledKernelRecord {
  KernelMetadata metadata;
  KernelEntry *entry = nullptr;
  std::vector<Complex64> matrix;
  std::optional<std::string> asm_text;
};

} // namespace cpu

namespace cuda {

struct KernelGenSpec {
  Precision precision = Precision::F64;
  double ztol = 0.0;
  double otol = 0.0;
  uint32_t sm_major = 0;
  uint32_t sm_minor = 0;
  uint32_t maxnreg = 0;
};

} // namespace cuda

} // namespace cast

#endif // CAST_SIMULATOR_SRC_CPP_INTERNAL_TYPES_HPP
