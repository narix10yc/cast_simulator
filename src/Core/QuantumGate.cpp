#include "cast/Core/QuantumGate.h"
#include "cast/ADT/ComplexSquareMatrix.h"
#include "cast/ADT/GateMatrix.h"
#include "utils/utils.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <iostream>

using namespace cast;

template <unsigned EdgeSize>
static ComplexSquareMatrix matmul_SameTargets(const ComplexSquareMatrix& A,
                                              const ComplexSquareMatrix& B) {
  assert(A.edgeSize() == EdgeSize);
  assert(B.edgeSize() == EdgeSize);

  auto* aRe = A.reData();
  auto* aIm = A.imData();
  auto* bRe = B.reData();
  auto* bIm = B.imData();
  ComplexSquareMatrix C(EdgeSize);
  auto* cRe = C.reData();
  auto* cIm = C.imData();
  for (uint64_t i = 0; i < EdgeSize; ++i) {
    for (uint64_t j = 0; j < EdgeSize; ++j) {
      double sumRe = 0.0;
      double sumIm = 0.0;
      for (uint64_t k = 0; k < EdgeSize; ++k) {
        double aRe_ik = aRe[i * EdgeSize + k];
        double aIm_ik = aIm[i * EdgeSize + k];
        double bRe_kj = bRe[k * EdgeSize + j];
        double bIm_kj = bIm[k * EdgeSize + j];
        sumRe += aRe_ik * bRe_kj - aIm_ik * bIm_kj;
        sumIm += aRe_ik * bIm_kj + aIm_ik * bRe_kj;
      }
      cRe[i * EdgeSize + j] = sumRe;
      cIm[i * EdgeSize + j] = sumIm;
    }
  }

  return C;
}

// Returns nullptr if this fast path is not applicable
static ScalarGateMatrixPtr
fastpath_SameTargets(const ScalarGateMatrix& scalarGM_A,
                     const ScalarGateMatrix& scalarGM_B) {
  assert(scalarGM_A.nQubits() == scalarGM_B.nQubits());
  const auto& A = scalarGM_A.matrix();
  const auto& B = scalarGM_B.matrix();

  switch (scalarGM_A.nQubits()) {
  case 1:
    return ScalarGateMatrix::Create(matmul_SameTargets<2>(A, B));
  case 2:
    return ScalarGateMatrix::Create(matmul_SameTargets<4>(A, B));
  case 3:
    return ScalarGateMatrix::Create(matmul_SameTargets<8>(A, B));
  default:
    break;
  }
  return nullptr;
}

// (I \otimes A) B
static ScalarGateMatrixPtr fastpath_a_ba(const ScalarGateMatrix& scalarGM_A,
                                         const ScalarGateMatrix& scalarGM_B) {
  assert(scalarGM_A.nQubits() == 1);
  assert(scalarGM_B.nQubits() == 2);

  auto* aRe = scalarGM_A.matrix().reData();
  auto* aIm = scalarGM_A.matrix().imData();
  auto* bRe = scalarGM_B.matrix().reData();
  auto* bIm = scalarGM_B.matrix().imData();

  return ScalarGateMatrix::Create(ComplexSquareMatrix{
      // clang-format off
  { // ---- Real ----
    aRe[0]*bRe[0]  - aIm[0]*bIm[0]  + aRe[1]*bRe[8]  - aIm[1]*bIm[8],
    aRe[0]*bRe[1]  - aIm[0]*bIm[1]  + aRe[1]*bRe[9]  - aIm[1]*bIm[9],
    aRe[0]*bRe[2]  - aIm[0]*bIm[2]  + aRe[1]*bRe[10] - aIm[1]*bIm[10],
    aRe[0]*bRe[3]  - aIm[0]*bIm[3]  + aRe[1]*bRe[11] - aIm[1]*bIm[11],

    aRe[0]*bRe[4]  - aIm[0]*bIm[4]  + aRe[1]*bRe[12] - aIm[1]*bIm[12],
    aRe[0]*bRe[5]  - aIm[0]*bIm[5]  + aRe[1]*bRe[13] - aIm[1]*bIm[13],
    aRe[0]*bRe[6]  - aIm[0]*bIm[6]  + aRe[1]*bRe[14] - aIm[1]*bIm[14],
    aRe[0]*bRe[7]  - aIm[0]*bIm[7]  + aRe[1]*bRe[15] - aIm[1]*bIm[15],

    aRe[2]*bRe[0]  - aIm[2]*bIm[0]  + aRe[3]*bRe[8]  - aIm[3]*bIm[8],
    aRe[2]*bRe[1]  - aIm[2]*bIm[1]  + aRe[3]*bRe[9]  - aIm[3]*bIm[9],
    aRe[2]*bRe[2]  - aIm[2]*bIm[2]  + aRe[3]*bRe[10] - aIm[3]*bIm[10],
    aRe[2]*bRe[3]  - aIm[2]*bIm[3]  + aRe[3]*bRe[11] - aIm[3]*bIm[11],

    aRe[2]*bRe[4]  - aIm[2]*bIm[4]  + aRe[3]*bRe[12] - aIm[3]*bIm[12],
    aRe[2]*bRe[5]  - aIm[2]*bIm[5]  + aRe[3]*bRe[13] - aIm[3]*bIm[13],
    aRe[2]*bRe[6]  - aIm[2]*bIm[6]  + aRe[3]*bRe[14] - aIm[3]*bIm[14],
    aRe[2]*bRe[7]  - aIm[2]*bIm[7]  + aRe[3]*bRe[15] - aIm[3]*bIm[15]
  },
  { // ---- Imag ----
    aRe[0]*bIm[0]  + aIm[0]*bRe[0]  + aRe[1]*bIm[8]  + aIm[1]*bRe[8],
    aRe[0]*bIm[1]  + aIm[0]*bRe[1]  + aRe[1]*bIm[9]  + aIm[1]*bRe[9],
    aRe[0]*bIm[2]  + aIm[0]*bRe[2]  + aRe[1]*bIm[10] + aIm[1]*bRe[10],
    aRe[0]*bIm[3]  + aIm[0]*bRe[3]  + aRe[1]*bIm[11] + aIm[1]*bRe[11],

    aRe[0]*bIm[4]  + aIm[0]*bRe[4]  + aRe[1]*bIm[12] + aIm[1]*bRe[12],
    aRe[0]*bIm[5]  + aIm[0]*bRe[5]  + aRe[1]*bIm[13] + aIm[1]*bRe[13],
    aRe[0]*bIm[6]  + aIm[0]*bRe[6]  + aRe[1]*bIm[14] + aIm[1]*bRe[14],
    aRe[0]*bIm[7]  + aIm[0]*bRe[7]  + aRe[1]*bIm[15] + aIm[1]*bRe[15],

    aRe[2]*bIm[0]  + aIm[2]*bRe[0]  + aRe[3]*bIm[8]  + aIm[3]*bRe[8],
    aRe[2]*bIm[1]  + aIm[2]*bRe[1]  + aRe[3]*bIm[9]  + aIm[3]*bRe[9],
    aRe[2]*bIm[2]  + aIm[2]*bRe[2]  + aRe[3]*bIm[10] + aIm[3]*bRe[10],
    aRe[2]*bIm[3]  + aIm[2]*bRe[3]  + aRe[3]*bIm[11] + aIm[3]*bRe[11],

    aRe[2]*bIm[4]  + aIm[2]*bRe[4]  + aRe[3]*bIm[12] + aIm[3]*bRe[12],
    aRe[2]*bIm[5]  + aIm[2]*bRe[5]  + aRe[3]*bIm[13] + aIm[3]*bRe[13],
    aRe[2]*bIm[6]  + aIm[2]*bRe[6]  + aRe[3]*bIm[14] + aIm[3]*bRe[14],
    aRe[2]*bIm[7]  + aIm[2]*bRe[7]  + aRe[3]*bIm[15] + aIm[3]*bRe[15]
  }
      // clang-format on
  });
}

// (A \otimes I) B
static ScalarGateMatrixPtr fastpath_b_ba(const ScalarGateMatrix& scalarGM_A,
                                         const ScalarGateMatrix& scalarGM_B) {
  assert(scalarGM_A.nQubits() == 1);
  assert(scalarGM_B.nQubits() == 2);

  auto* aRe = scalarGM_A.matrix().reData();
  auto* aIm = scalarGM_A.matrix().imData();
  auto* bRe = scalarGM_B.matrix().reData();
  auto* bIm = scalarGM_B.matrix().imData();

  return ScalarGateMatrix::Create(ComplexSquareMatrix{
      // clang-format off
  { // ---- Real (row-major) ----
    aRe[0]*bRe[0]  - aIm[0]*bIm[0]  + aRe[1]*bRe[4]  - aIm[1]*bIm[4],
    aRe[0]*bRe[1]  - aIm[0]*bIm[1]  + aRe[1]*bRe[5]  - aIm[1]*bIm[5],
    aRe[0]*bRe[2]  - aIm[0]*bIm[2]  + aRe[1]*bRe[6]  - aIm[1]*bIm[6],
    aRe[0]*bRe[3]  - aIm[0]*bIm[3]  + aRe[1]*bRe[7]  - aIm[1]*bIm[7],

    aRe[2]*bRe[0]  - aIm[2]*bIm[0]  + aRe[3]*bRe[4]  - aIm[3]*bIm[4],
    aRe[2]*bRe[1]  - aIm[2]*bIm[1]  + aRe[3]*bRe[5]  - aIm[3]*bIm[5],
    aRe[2]*bRe[2]  - aIm[2]*bIm[2]  + aRe[3]*bRe[6]  - aIm[3]*bIm[6],
    aRe[2]*bRe[3]  - aIm[2]*bIm[3]  + aRe[3]*bRe[7]  - aIm[3]*bIm[7],

    aRe[0]*bRe[8]  - aIm[0]*bIm[8]  + aRe[1]*bRe[12] - aIm[1]*bIm[12],
    aRe[0]*bRe[9]  - aIm[0]*bIm[9]  + aRe[1]*bRe[13] - aIm[1]*bIm[13],
    aRe[0]*bRe[10] - aIm[0]*bIm[10] + aRe[1]*bRe[14] - aIm[1]*bIm[14],
    aRe[0]*bRe[11] - aIm[0]*bIm[11] + aRe[1]*bRe[15] - aIm[1]*bIm[15],

    aRe[2]*bRe[8]  - aIm[2]*bIm[8]  + aRe[3]*bRe[12] - aIm[3]*bIm[12],
    aRe[2]*bRe[9]  - aIm[2]*bIm[9]  + aRe[3]*bRe[13] - aIm[3]*bIm[13],
    aRe[2]*bRe[10] - aIm[2]*bIm[10] + aRe[3]*bRe[14] - aIm[3]*bIm[14],
    aRe[2]*bRe[11] - aIm[2]*bIm[11] + aRe[3]*bRe[15] - aIm[3]*bIm[15]
  },
  { // ---- Imag ----
    aRe[0]*bIm[0]  + aIm[0]*bRe[0]  + aRe[1]*bIm[4]  + aIm[1]*bRe[4],
    aRe[0]*bIm[1]  + aIm[0]*bRe[1]  + aRe[1]*bIm[5]  + aIm[1]*bRe[5],
    aRe[0]*bIm[2]  + aIm[0]*bRe[2]  + aRe[1]*bIm[6]  + aIm[1]*bRe[6],
    aRe[0]*bIm[3]  + aIm[0]*bRe[3]  + aRe[1]*bIm[7]  + aIm[1]*bRe[7],

    aRe[2]*bIm[0]  + aIm[2]*bRe[0]  + aRe[3]*bIm[4]  + aIm[3]*bRe[4],
    aRe[2]*bIm[1]  + aIm[2]*bRe[1]  + aRe[3]*bIm[5]  + aIm[3]*bRe[5],
    aRe[2]*bIm[2]  + aIm[2]*bRe[2]  + aRe[3]*bIm[6]  + aIm[3]*bRe[6],
    aRe[2]*bIm[3]  + aIm[2]*bRe[3]  + aRe[3]*bIm[7]  + aIm[3]*bRe[7],

    aRe[0]*bIm[8]  + aIm[0]*bRe[8]  + aRe[1]*bIm[12] + aIm[1]*bRe[12],
    aRe[0]*bIm[9]  + aIm[0]*bRe[9]  + aRe[1]*bIm[13] + aIm[1]*bRe[13],
    aRe[0]*bIm[10] + aIm[0]*bRe[10] + aRe[1]*bIm[14] + aIm[1]*bRe[14],
    aRe[0]*bIm[11] + aIm[0]*bRe[11] + aRe[1]*bIm[15] + aIm[1]*bRe[15],

    aRe[2]*bIm[8]  + aIm[2]*bRe[8]  + aRe[3]*bIm[12] + aIm[3]*bRe[12],
    aRe[2]*bIm[9]  + aIm[2]*bRe[9]  + aRe[3]*bIm[13] + aIm[3]*bRe[13],
    aRe[2]*bIm[10] + aIm[2]*bRe[10] + aRe[3]*bIm[14] + aIm[3]*bRe[14],
    aRe[2]*bIm[11] + aIm[2]*bRe[11] + aRe[3]*bIm[15] + aIm[3]*bRe[15]
  }
      // clang-format on
  });
}

// A (I \otimes B)
static ScalarGateMatrixPtr fastpath_ba_a(const ScalarGateMatrix& scalarGM_A,
                                         const ScalarGateMatrix& scalarGM_B) {
  assert(scalarGM_A.nQubits() == 2);
  assert(scalarGM_B.nQubits() == 1);

  auto* aRe = scalarGM_A.matrix().reData();
  auto* aIm = scalarGM_A.matrix().imData();
  auto* bRe = scalarGM_B.matrix().reData();
  auto* bIm = scalarGM_B.matrix().imData();

  return ScalarGateMatrix::Create(ComplexSquareMatrix{
      // clang-format off
  { // -------- Real (row-major) --------
    // row 0
    aRe[0]*bRe[0] - aIm[0]*bIm[0] + aRe[1]*bRe[2] - aIm[1]*bIm[2],
    aRe[0]*bRe[1] - aIm[0]*bIm[1] + aRe[1]*bRe[3] - aIm[1]*bIm[3],
    aRe[2]*bRe[0] - aIm[2]*bIm[0] + aRe[3]*bRe[2] - aIm[3]*bIm[2],
    aRe[2]*bRe[1] - aIm[2]*bIm[1] + aRe[3]*bRe[3] - aIm[3]*bIm[3],
    // row 1
    aRe[4]*bRe[0] - aIm[4]*bIm[0] + aRe[5]*bRe[2] - aIm[5]*bIm[2],
    aRe[4]*bRe[1] - aIm[4]*bIm[1] + aRe[5]*bRe[3] - aIm[5]*bIm[3],
    aRe[6]*bRe[0] - aIm[6]*bIm[0] + aRe[7]*bRe[2] - aIm[7]*bIm[2],
    aRe[6]*bRe[1] - aIm[6]*bIm[1] + aRe[7]*bRe[3] - aIm[7]*bIm[3],
    // row 2
    aRe[8]*bRe[0] - aIm[8]*bIm[0] + aRe[9]*bRe[2] - aIm[9]*bIm[2],
    aRe[8]*bRe[1] - aIm[8]*bIm[1] + aRe[9]*bRe[3] - aIm[9]*bIm[3],
    aRe[10]*bRe[0] - aIm[10]*bIm[0] + aRe[11]*bRe[2] - aIm[11]*bIm[2],
    aRe[10]*bRe[1] - aIm[10]*bIm[1] + aRe[11]*bRe[3] - aIm[11]*bIm[3],
    // row 3
    aRe[12]*bRe[0] - aIm[12]*bIm[0] + aRe[13]*bRe[2] - aIm[13]*bIm[2],
    aRe[12]*bRe[1] - aIm[12]*bIm[1] + aRe[13]*bRe[3] - aIm[13]*bIm[3],
    aRe[14]*bRe[0] - aIm[14]*bIm[0] + aRe[15]*bRe[2] - aIm[15]*bIm[2],
    aRe[14]*bRe[1] - aIm[14]*bIm[1] + aRe[15]*bRe[3] - aIm[15]*bIm[3]
  },
  { // -------- Imag --------
    // row 0
    aRe[0]*bIm[0] + aIm[0]*bRe[0] + aRe[1]*bIm[2] + aIm[1]*bRe[2],
    aRe[0]*bIm[1] + aIm[0]*bRe[1] + aRe[1]*bIm[3] + aIm[1]*bRe[3],
    aRe[2]*bIm[0] + aIm[2]*bRe[0] + aRe[3]*bIm[2] + aIm[3]*bRe[2],
    aRe[2]*bIm[1] + aIm[2]*bRe[1] + aRe[3]*bIm[3] + aIm[3]*bRe[3],
    // row 1
    aRe[4]*bIm[0] + aIm[4]*bRe[0] + aRe[5]*bIm[2] + aIm[5]*bRe[2],
    aRe[4]*bIm[1] + aIm[4]*bRe[1] + aRe[5]*bIm[3] + aIm[5]*bRe[3],
    aRe[6]*bIm[0] + aIm[6]*bRe[0] + aRe[7]*bIm[2] + aIm[7]*bRe[2],
    aRe[6]*bIm[1] + aIm[6]*bRe[1] + aRe[7]*bIm[3] + aIm[7]*bRe[3],
    // row 2
    aRe[8]*bIm[0] + aIm[8]*bRe[0] + aRe[9]*bIm[2] + aIm[9]*bRe[2],
    aRe[8]*bIm[1] + aIm[8]*bRe[1] + aRe[9]*bIm[3] + aIm[9]*bRe[3],
    aRe[10]*bIm[0] + aIm[10]*bRe[0] + aRe[11]*bIm[2] + aIm[11]*bRe[2],
    aRe[10]*bIm[1] + aIm[10]*bRe[1] + aRe[11]*bIm[3] + aIm[11]*bRe[3],
    // row 3
    aRe[12]*bIm[0] + aIm[12]*bRe[0] + aRe[13]*bIm[2] + aIm[13]*bRe[2],
    aRe[12]*bIm[1] + aIm[12]*bRe[1] + aRe[13]*bIm[3] + aIm[13]*bRe[3],
    aRe[14]*bIm[0] + aIm[14]*bRe[0] + aRe[15]*bIm[2] + aIm[15]*bRe[2],
    aRe[14]*bIm[1] + aIm[14]*bRe[1] + aRe[15]*bIm[3] + aIm[15]*bRe[3]
  }
      // clang-format on
  });
}

// A (B \otimes I)
static ScalarGateMatrixPtr fastpath_ba_b(const ScalarGateMatrix& scalarGM_A,
                                         const ScalarGateMatrix& scalarGM_B) {
  assert(scalarGM_A.nQubits() == 2);
  assert(scalarGM_B.nQubits() == 1);

  auto* aRe = scalarGM_A.matrix().reData();
  auto* aIm = scalarGM_A.matrix().imData();
  auto* bRe = scalarGM_B.matrix().reData();
  auto* bIm = scalarGM_B.matrix().imData();

  return ScalarGateMatrix::Create(ComplexSquareMatrix{
      // clang-format off
    { // -------- Real (row-major) --------
    // row 0
    aRe[0]*bRe[0] - aIm[0]*bIm[0] + aRe[2]*bRe[2] - aIm[2]*bIm[2],
    aRe[1]*bRe[0] - aIm[1]*bIm[0] + aRe[3]*bRe[2] - aIm[3]*bIm[2],
    aRe[0]*bRe[1] - aIm[0]*bIm[1] + aRe[2]*bRe[3] - aIm[2]*bIm[3],
    aRe[1]*bRe[1] - aIm[1]*bIm[1] + aRe[3]*bRe[3] - aIm[3]*bIm[3],
    // row 1
    aRe[4]*bRe[0] - aIm[4]*bIm[0] + aRe[6]*bRe[2] - aIm[6]*bIm[2],
    aRe[5]*bRe[0] - aIm[5]*bIm[0] + aRe[7]*bRe[2] - aIm[7]*bIm[2],
    aRe[4]*bRe[1] - aIm[4]*bIm[1] + aRe[6]*bRe[3] - aIm[6]*bIm[3],
    aRe[5]*bRe[1] - aIm[5]*bIm[1] + aRe[7]*bRe[3] - aIm[7]*bIm[3],
    // row 2
    aRe[8]*bRe[0] - aIm[8]*bIm[0] + aRe[10]*bRe[2] - aIm[10]*bIm[2],
    aRe[9]*bRe[0] - aIm[9]*bIm[0] + aRe[11]*bRe[2] - aIm[11]*bIm[2],
    aRe[8]*bRe[1] - aIm[8]*bIm[1] + aRe[10]*bRe[3] - aIm[10]*bIm[3],
    aRe[9]*bRe[1] - aIm[9]*bIm[1] + aRe[11]*bRe[3] - aIm[11]*bIm[3],
    // row 3
    aRe[12]*bRe[0] - aIm[12]*bIm[0] + aRe[14]*bRe[2] - aIm[14]*bIm[2],
    aRe[13]*bRe[0] - aIm[13]*bIm[0] + aRe[15]*bRe[2] - aIm[15]*bIm[2],
    aRe[12]*bRe[1] - aIm[12]*bIm[1] + aRe[14]*bRe[3] - aIm[14]*bIm[3],
    aRe[13]*bRe[1] - aIm[13]*bIm[1] + aRe[15]*bRe[3] - aIm[15]*bIm[3]
  },
  { // -------- Imag --------
    // row 0
    aRe[0]*bIm[0] + aIm[0]*bRe[0] + aRe[2]*bIm[2] + aIm[2]*bRe[2],
    aRe[1]*bIm[0] + aIm[1]*bRe[0] + aRe[3]*bIm[2] + aIm[3]*bRe[2],
    aRe[0]*bIm[1] + aIm[0]*bRe[1] + aRe[2]*bIm[3] + aIm[2]*bRe[3],
    aRe[1]*bIm[1] + aIm[1]*bRe[1] + aRe[3]*bIm[3] + aIm[3]*bRe[3],
    // row 1
    aRe[4]*bIm[0] + aIm[4]*bRe[0] + aRe[6]*bIm[2] + aIm[6]*bRe[2],
    aRe[5]*bIm[0] + aIm[5]*bRe[0] + aRe[7]*bIm[2] + aIm[7]*bRe[2],
    aRe[4]*bIm[1] + aIm[4]*bRe[1] + aRe[6]*bIm[3] + aIm[6]*bRe[3],
    aRe[5]*bIm[1] + aIm[5]*bRe[1] + aRe[7]*bIm[3] + aIm[7]*bRe[3],
    // row 2
    aRe[8]*bIm[0] + aIm[8]*bRe[0] + aRe[10]*bIm[2] + aIm[10]*bRe[2],
    aRe[9]*bIm[0] + aIm[9]*bRe[0] + aRe[11]*bIm[2] + aIm[11]*bRe[2],
    aRe[8]*bIm[1] + aIm[8]*bRe[1] + aRe[10]*bIm[3] + aIm[10]*bRe[3],
    aRe[9]*bIm[1] + aIm[9]*bRe[1] + aRe[11]*bIm[3] + aIm[11]*bRe[3],
    // row 3
    aRe[12]*bIm[0] + aIm[12]*bRe[0] + aRe[14]*bIm[2] + aIm[14]*bRe[2],
    aRe[13]*bIm[0] + aIm[13]*bRe[0] + aRe[15]*bIm[2] + aIm[15]*bRe[2],
    aRe[12]*bIm[1] + aIm[12]*bRe[1] + aRe[14]*bIm[3] + aIm[14]*bRe[3],
    aRe[13]*bIm[1] + aIm[13]*bRe[1] + aRe[15]*bIm[3] + aIm[15]*bRe[3]
  }
      // clang-format on
  });
}

/**** Matmul of Quantum Gates ****/
QuantumGatePtr cast::matmul(const QuantumGate* gateA,
                            const QuantumGate* gateB) {
  assert(gateA != nullptr);
  assert(gateB != nullptr);

  const auto* aStdQuGate = llvm::dyn_cast<StandardQuantumGate>(gateA);
  const auto* bStdQuGate = llvm::dyn_cast<StandardQuantumGate>(gateB);

  assert(aStdQuGate && bStdQuGate &&
         "Only implemented matmul for StandardQuantumGate now");

  assert(aStdQuGate->noiseChannel() == nullptr &&
         bStdQuGate->noiseChannel() == nullptr &&
         "Only implemented matmul for noiseless gates now");

  const auto aScalarGM = aStdQuGate->getScalarGM();
  const auto bScalarGM = bStdQuGate->getScalarGM();
  assert(aScalarGM && bScalarGM &&
         "Both gate matrices must be ScalarGateMatrix for now");

  // fast path: same target qubits
  if (aStdQuGate->qubits() == bStdQuGate->qubits()) {
    // cScalarGM can be nullptr when nQubits() is large
    auto cScalarGM = fastpath_SameTargets(*aScalarGM, *bScalarGM);
    if (cScalarGM != nullptr) {
      return StandardQuantumGate::Create(
          cScalarGM, NoiseChannelPtr(nullptr), aStdQuGate->qubits());
    }
  }

  // fast path: A acts on one qubit, B acts on two qubits
  if (aStdQuGate->nQubits() == 1 && bStdQuGate->nQubits() == 2) {
    // check if A's qubit is one of B's qubits
    const int aQubit = aStdQuGate->qubits()[0];
    const auto& bQubits = bStdQuGate->qubits();
    if (aQubit == bQubits[0]) {
      auto cScalarGM = fastpath_a_ba(*aScalarGM, *bScalarGM);
      assert(cScalarGM != nullptr);
      return StandardQuantumGate::Create(
          cScalarGM, NoiseChannelPtr(nullptr), bStdQuGate->qubits());
    } else if (aQubit == bQubits[1]) {
      auto cScalarGM = fastpath_b_ba(*aScalarGM, *bScalarGM);
      assert(cScalarGM != nullptr);
      return StandardQuantumGate::Create(
          cScalarGM, NoiseChannelPtr(nullptr), bStdQuGate->qubits());
    }
  }

  // fast path: A acts on two qubits, B acts on one qubit
  if (aStdQuGate->nQubits() == 2 && bStdQuGate->nQubits() == 1) {
    // check if B's qubit is one of A's qubits
    const int bQubit = bStdQuGate->qubits()[0];
    const auto& aQubits = aStdQuGate->qubits();
    if (bQubit == aQubits[0]) {
      auto cScalarGM = fastpath_ba_a(*aScalarGM, *bScalarGM);
      assert(cScalarGM != nullptr);
      return StandardQuantumGate::Create(
          cScalarGM, NoiseChannelPtr(nullptr), aStdQuGate->qubits());
    } else if (bQubit == aQubits[1]) {
      auto cScalarGM = fastpath_ba_b(*aScalarGM, *bScalarGM);
      assert(cScalarGM != nullptr);
      return StandardQuantumGate::Create(
          cScalarGM, NoiseChannelPtr(nullptr), aStdQuGate->qubits());
    }
  }

  // general case
  // C = AB
  const auto& aQubits = gateA->qubits();
  const auto& bQubits = gateB->qubits();

  const int anQubits = aQubits.size();
  const int bnQubits = bQubits.size();

  struct TargetQubitsInfo {
    int q;    // the qubit
    int aIdx; // index in aQubits, -1 if not in aQubits
    int bIdx; // index in bQubits, -1 if not in bQubits
  };

  // setup result gate target qubits
  std::vector<TargetQubitsInfo> targetQubitsInfo;
  targetQubitsInfo.reserve(anQubits + bnQubits);
  {
    int aIdx = 0, bIdx = 0;
    while (aIdx < anQubits || bIdx < bnQubits) {
      if (aIdx == anQubits) {
        for (; bIdx < bnQubits; ++bIdx)
          targetQubitsInfo.emplace_back(bQubits[bIdx], -1, bIdx);
        break;
      }
      if (bIdx == bnQubits) {
        for (; aIdx < anQubits; ++aIdx)
          targetQubitsInfo.emplace_back(aQubits[aIdx], aIdx, -1);
        break;
      }
      int aQubit = aQubits[aIdx];
      int bQubit = bQubits[bIdx];
      if (aQubit == bQubit) {
        targetQubitsInfo.emplace_back(aQubit, aIdx++, bIdx++);
        continue;
      }
      if (aQubit < bQubit) {
        targetQubitsInfo.emplace_back(aQubit, aIdx++, -1);
        continue;
      }
      // otherwise, aQubit > bQubit
      targetQubitsInfo.emplace_back(bQubit, -1, bIdx++);
    }
    assert(aIdx == anQubits && bIdx == bnQubits);
  }

  const int cnQubits = targetQubitsInfo.size();
  uint64_t aPextMask = 0ULL;
  uint64_t bPextMask = 0ULL;
  uint64_t aZeroingMask = ~0ULL;
  uint64_t bZeroingMask = ~0ULL;
  std::vector<uint64_t> aSharedQubitShifts, bSharedQubitShifts;
  for (unsigned i = 0; i < cnQubits; ++i) {
    const int aIdx = targetQubitsInfo[i].aIdx;
    const int bIdx = targetQubitsInfo[i].bIdx;
    // shared qubit: set zeroing masks and shifts
    if (aIdx >= 0 && bIdx >= 0) {
      aZeroingMask ^= (1ULL << aIdx);
      bZeroingMask ^= (1ULL << (bIdx + bnQubits));
      aSharedQubitShifts.emplace_back(1ULL << aIdx);
      bSharedQubitShifts.emplace_back(1ULL << (bIdx + bnQubits));
    }

    if (aIdx >= 0)
      aPextMask |= (1 << i) + (1 << (i + cnQubits));

    if (bIdx >= 0)
      bPextMask |= (1 << i) + (1 << (i + cnQubits));
  }
  const int contractionWidth = aSharedQubitShifts.size();
  std::vector<int> cQubits;
  cQubits.reserve(cnQubits);
  for (const auto& tQubit : targetQubitsInfo)
    cQubits.push_back(tQubit.q);

  // std::cerr << CYAN_FG << "Debug:\n";
  // utils::printArray(std::cerr << "aQubits: ", aQubits) << "\n";
  // utils::printArray(std::cerr << "bQubits: ", bQubits) << "\n";
  // utils::printArray(std::cerr << "target qubits (q, aIdx, bIdx): ",
  //   llvm::ArrayRef(targetQubitsInfo),
  //   [](std::ostream& os, const TargetQubitsInfo& q) {
  //     os << "(" << q.q << "," << q.aIdx << "," << q.bIdx << ")";
  //   }) << "\n";

  // std::cerr << "aPextMask: " << utils::as0b(aPextMask, 10) << "\n"
  //           << "bPextMask: " << utils::as0b(bPextMask, 10) << "\n"
  //           << "aZeroingMask: " << utils::as0b(aZeroingMask, 10) << "\n"
  //           << "bZeroingMask: " << utils::as0b(bZeroingMask, 10) << "\n";
  // utils::printArray(std::cerr << "a shifts: ", aSharedQubitShifts) << "\n";
  // utils::printArray(std::cerr << "b shifts: ", bSharedQubitShifts) << "\n";
  // std::cerr << "contraction width = " << contractionWidth << "\n";
  // std::cerr << RESET;

  const auto matmulComplexSquareMatrix =
      [&, cnQubits = cnQubits](const ComplexSquareMatrix& matA,
                               const ComplexSquareMatrix& matB,
                               ComplexSquareMatrix& matC) -> void {
    for (uint64_t cIdx = 0ULL; cIdx < matC.halfSize(); ++cIdx) {
      uint64_t aIdxBegin =
          utils::pext64(cIdx, aPextMask, 2 * cnQubits) & aZeroingMask;
      uint64_t bIdxBegin =
          utils::pext64(cIdx, bPextMask, 2 * cnQubits) & bZeroingMask;

      // std::cerr << "Ready to update cmat[" << i
      //           << " (" << utils::as0b(i, 2 * cnQubits) << ")]\n"
      //           << "  aIdxBegin: " << utils::as0b(i, 2 * cnQubits)
      //           << " -> " << utils::pext64(i, aPextMask) << " ("
      //           << utils::as0b(utils::pext64(i, aPextMask), 2 * anQubits)
      //           << ") -> " << aIdxBegin << " ("
      //           << utils::as0b(aIdxBegin, 2 * anQubits) << ")\n"
      //           << "  bIdxBegin: " << utils::as0b(i, 2 * cnQubits)
      //           << " -> "
      //           << utils::as0b(utils::pext64(i, bPextMask), 2 * bnQubits)
      //           << " -> " << utils::as0b(bIdxBegin, 2 * bnQubits)
      //           << " = " << bIdxBegin << "\n";
      matC.reData()[cIdx] = 0.0;
      matC.imData()[cIdx] = 0.0;
      for (uint64_t s = 0; s < (1ULL << contractionWidth); ++s) {
        uint64_t aIdx = aIdxBegin;
        uint64_t bIdx = bIdxBegin;
        for (unsigned bit = 0; bit < contractionWidth; ++bit) {
          if (s & (1 << bit)) {
            aIdx += aSharedQubitShifts[bit];
            bIdx += bSharedQubitShifts[bit];
          }
        }
        // std::cerr << "  aIdx = " << aIdx << ": " << aCMat->data()[aIdx] <<
        // ";"
        // << "  bIdx = " << bIdx << ": " << bCMat->data()[bIdx] << "\n";
        matC.reData()[cIdx] += matA.reData()[aIdx] * matB.reData()[bIdx] -
                               matA.imData()[aIdx] * matB.imData()[bIdx];
        matC.imData()[cIdx] += matA.reData()[aIdx] * matB.imData()[bIdx] +
                               matA.imData()[aIdx] * matB.reData()[bIdx];
      }
    }
  };

  const ComplexSquareMatrix& aCMat = aScalarGM->matrix();
  const ComplexSquareMatrix& bCMat = bScalarGM->matrix();
  ComplexSquareMatrix cCMat(1ULL << cnQubits);
  matmulComplexSquareMatrix(aCMat, bCMat, cCMat);

  return StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(std::move(cCMat)),
      NoiseChannelPtr(nullptr), // No noise channel for now
      cQubits);

  assert(false && "Matmul not implemented for these gate types yet");
  return nullptr;
}

SuperopQuantumGatePtr StandardQuantumGate::getSuperopGate() const {
  assert(gateMatrix_ != nullptr);
  auto scalarGM = getScalarGM();
  assert(scalarGM != nullptr && "Only supporting ScalarGateMatrix for now");
  if (noiseChannel_ == nullptr) {
    // If there is no noise channel, the superoperator matrix is just the gate
    // matrix itself.
    auto superopGM = std::make_shared<ScalarGateMatrix>(scalarGM->matrix());
    return SuperopQuantumGate::Create(superopGM, qubits());
  }

  assert(noiseChannel_->reps.krausRep != nullptr &&
         "We must have KrausRep to compute superoperator matrix");
  const auto& krausOps = noiseChannel_->reps.krausRep->getOps();
  // If the gate matrix and every Kraus operator are N * N, then the
  // superoperator matrix is (N ** 2) * (N ** 2).
  const size_t N = 1ULL << nQubits();
  auto superopGM = std::make_shared<ScalarGateMatrix>(nQubits() * 2);
  auto& superopMatrix = superopGM->matrix();
  superopMatrix.fillZeros();
  for (const auto& krausOp : krausOps) {
    // compute the new Kraus operator F_k = E_k U
    auto newKrausOp = std::make_shared<ScalarGateMatrix>(nQubits());
    cast::matmul(krausOp.matrix(), scalarGM->matrix(), newKrausOp->matrix());
    const auto& newKrausOpM = newKrausOp->matrix();

    // compute the tensor product F_K.conj() \otimes F_k
    for (size_t r = 0; r < N; ++r) {
      for (size_t c = 0; c < N; ++c) {
        for (size_t rr = 0; rr < N; ++rr) {
          for (size_t cc = 0; cc < N; ++cc) {
            size_t row = r * N + rr;
            size_t col = c * N + cc;
            // (re(r, c) - i * im(r, c)) * (re(rr, cc) + i * im(rr, cc))
            // real part is re(r, c) * re(rr, cc) + im(r, c) * im(rr, cc)
            // imag part is re(r, c) * im(rr, cc) - im(r, c) * re(rr, cc)
            superopMatrix.real(row, col) +=
                newKrausOpM.real(r, c) * newKrausOpM.real(rr, cc) +
                newKrausOpM.imag(r, c) * newKrausOpM.imag(rr, cc);
            superopMatrix.imag(row, col) +=
                newKrausOpM.real(r, c) * newKrausOpM.imag(rr, cc) -
                newKrausOpM.imag(r, c) * newKrausOpM.real(rr, cc);
          }
        }
      }
    }
  }
  return SuperopQuantumGate::Create(superopGM, qubits());
}

/**** op count *****/
namespace {
size_t countNonZeroElems(const ScalarGateMatrix& matrix, double zeroTol) {
  if (zeroTol <= 0.0)
    return matrix.matrix().size();

  size_t count = 0;
  size_t len = matrix.matrix().size();
  const auto* data = matrix.matrix().data();
  for (size_t i = 0; i < len; ++i) {
    if (std::abs(data[i]) > zeroTol)
      ++count;
  }
  return count;
}
} // anonymous namespace

double StandardQuantumGate::opCount(double zeroTol) const {
  auto scalarGM = getScalarGM();
  assert(scalarGM != nullptr && "Only supporting scalar gate matrix for now");

  if (noiseChannel_ == nullptr) {
    double count = static_cast<double>(countNonZeroElems(*scalarGM, zeroTol));
    return count * std::pow<double>(2.0, 1 - nQubits());
  }

  // If there is a noise channel, we need to compute the superoperator matrix
  // and count the non-zero elements in it.
  auto superopGate = getSuperopGate();
  assert(superopGate != nullptr && "Superop gate should not be null");
  return superopGate->opCount(zeroTol);
}

double SuperopQuantumGate::opCount(double zeroTol) const {
  assert(superopMatrix_ != nullptr && "Superop matrix is null");
  double count =
      static_cast<double>(countNonZeroElems(*superopMatrix_, zeroTol));

  // superop matrices are treated as 2n-qubit gates
  return count * std::pow<double>(2.0, 1 - 2 * nQubits());
}

/**** Inverse ****/

QuantumGatePtr StandardQuantumGate::inverse() const {
  if (noiseChannel_ != nullptr)
    return nullptr;
  auto scalarGM = getScalarGM();
  if (scalarGM == nullptr)
    return nullptr;
  ComplexSquareMatrix matinv(scalarGM->matrix().edgeSize());
  if (!cast::matinv(scalarGM->matrix(), matinv))
    return nullptr;
  return StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(std::move(matinv)),
      nullptr, // No noise channel
      qubits());
}

/**** is commuting ****/

bool cast::isCommuting(const QuantumGate* gateA,
                       const QuantumGate* gateB,
                       double tol) {
  if (gateA == nullptr || gateB == nullptr) {
    assert(false && "One of the gates is null");
    return false;
  }

  auto gateAB = matmul(gateA, gateB);
  if (gateAB == nullptr) {
    assert(false && "Failed to compute AB");
    return false;
  }
  auto gateBA = matmul(gateB, gateA);
  if (gateBA == nullptr) {
    assert(false && "Failed to compute BA");
    return false;
  }

  auto* stdGateAB = llvm::dyn_cast<StandardQuantumGate>(gateAB.get());
  auto* stdGateBA = llvm::dyn_cast<StandardQuantumGate>(gateBA.get());
  if (stdGateAB == nullptr || stdGateBA == nullptr) {
    assert(false && "Both gates must be StandardQuantumGate for now");
    return false;
  }
  auto scalarGM_AB = stdGateAB->getScalarGM();
  auto scalarGM_BA = stdGateBA->getScalarGM();
  auto diff = cast::maximum_norm(scalarGM_AB->matrix(), scalarGM_BA->matrix());
  return diff <= tol;
}
