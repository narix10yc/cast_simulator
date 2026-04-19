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

inline bool isValidPrecision(Precision p) { return p == Precision::F32 || p == Precision::F64; }

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
  SimdWidth simdWidth = SimdWidth::W256;
  MatrixLoadMode mode = MatrixLoadMode::ImmValue;
  double ztol = 0.0;
  double otol = 0.0;
};

struct KernelMetadata {
  KernelId kernelId = 0;
  Precision precision = Precision::F64;
  SimdWidth simdWidth = SimdWidth::W256;
  MatrixLoadMode mode = MatrixLoadMode::ImmValue;
  uint32_t nGateQubits = 0;
};

struct CompiledKernelRecord {
  KernelMetadata metadata;
  KernelEntry *entry = nullptr;
  std::vector<Complex64> matrix;
  std::optional<std::string> irText;
  std::optional<std::string> asmText;
};

} // namespace cpu

namespace cuda {

struct KernelGenSpec {
  Precision precision = Precision::F64;
  double ztol = 0.0;
  double otol = 0.0;
  uint32_t smMajor = 0;
  uint32_t smMinor = 0;
  uint32_t maxNReg = 0;
};

} // namespace cuda

} // namespace cast

#endif // CAST_SIMULATOR_SRC_CPP_INTERNAL_TYPES_HPP
