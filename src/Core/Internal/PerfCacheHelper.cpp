#include "cast/Internal/PerfCacheHelper.h"
#include "llvm/Support/Casting.h"
#include <random>

using namespace cast;

double cast::internal::calculateMemUpdateSpeed(int nQubits,
                                               Precision precision,
                                               double t) {
  assert(nQubits > 0);
  assert(precision == Precision::FP32 || precision == Precision::FP64);
  assert(t > 0.0);

  // FP32 takes 8 bytes because of complex number
  uint64_t dataSize = (precision == Precision::FP32 ? 8ULL : 16ULL) << nQubits;
  return static_cast<double>(dataSize) * 1e-9 / t;
}

void cast::internal::randRemoveQuantumGate(QuantumGate* quGate, float p) {
  assert(0.0f <= p && p <= 1.0f);
  if (p == 0.0f)
    return; // nothing to do

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distF(0.0f, 1.0f);

  auto* stdQuGate = llvm::dyn_cast<StandardQuantumGate>(quGate);
  assert(stdQuGate != nullptr && "Only implemented for StandardQuantumGate");
  auto scalarGM = stdQuGate->getScalarGM();
  assert(scalarGM != nullptr);

  auto& mat = scalarGM->matrix();
  auto edgeSize = mat.edgeSize();

  // randomly zero out some elements
  for (unsigned r = 0; r < edgeSize; ++r) {
    for (unsigned c = 0; c < edgeSize; ++c) {
      if (distF(gen) < p) {
        // zero out the element
        mat.reData()[r * edgeSize + c] = 0.0;
        mat.imData()[r * edgeSize + c] = 0.0;
      }
    }
  }

  std::uniform_int_distribution<unsigned> distI(0, edgeSize - 1);
  // check if some row is completely zeroed out
  for (unsigned r = 0; r < edgeSize; ++r) {
    bool isRowZeroed = true;
    for (unsigned c = 0; c < edgeSize; ++c) {
      if (mat.reData()[r * edgeSize + c] != 0.0 ||
          mat.imData()[r * edgeSize + c] != 0.0) {
        isRowZeroed = false;
        break;
      }
    }
    if (isRowZeroed) {
      // randomly choose a non-zero element to keep
      auto keepCol = distI(gen);
      mat.reData()[r * edgeSize + keepCol] = 0.5;
      mat.imData()[r * edgeSize + keepCol] = 0.5;
    }
  }

  // check if some column is completely zeroed out
  for (unsigned c = 0; c < edgeSize; ++c) {
    bool isColZeroed = true;
    for (unsigned r = 0; r < edgeSize; ++r) {
      if (mat.reData()[r * edgeSize + c] != 0.0 ||
          mat.imData()[r * edgeSize + c] != 0.0) {
        isColZeroed = false;
        break;
      }
    }
    if (isColZeroed) {
      // randomly choose a non-zero element to keep
      auto keepRow = distI(gen);
      mat.reData()[keepRow * edgeSize + c] = 0.5;
      mat.imData()[keepRow * edgeSize + c] = 0.5;
    }
  }
}