#include "cast/Core/QuantumGate.h"
#include "utils/iocolor.h"
#include "utils/utils.h"
#include "utils/PrintSpan.h"
#include "llvm/Support/Casting.h"

using namespace cast;

/**** Matmul of Quantum Gates ****/
QuantumGatePtr cast::matmul(const QuantumGate* gateA,
                            const QuantumGate* gateB) {
  assert(gateA != nullptr);
  assert(gateB != nullptr);
  // C = AB
  const auto& aQubits = gateA->qubits();
  const auto& bQubits = gateB->qubits();
  const int anQubits = aQubits.size();
  const int bnQubits = bQubits.size();

  struct TargetQubitsInfo {
    int q; // the qubit
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


  const auto matmulComplexSquareMatrix = [&, cnQubits=cnQubits](
      const ComplexSquareMatrix& matA,
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
        // std::cerr << "  aIdx = " << aIdx << ": " << aCMat->data()[aIdx] << ";"
                  // << "  bIdx = " << bIdx << ": " << bCMat->data()[bIdx] << "\n";
        matC.reData()[cIdx] += matA.reData()[aIdx] * matB.reData()[bIdx] -
                               matA.imData()[aIdx] * matB.imData()[bIdx];
        matC.imData()[cIdx] += matA.reData()[aIdx] * matB.imData()[bIdx] +
                               matA.imData()[aIdx] * matB.reData()[bIdx];
      }
    }
  };

  const auto* aStdQuGate = llvm::dyn_cast<StandardQuantumGate>(gateA);
  const auto* bStdQuGate = llvm::dyn_cast<StandardQuantumGate>(gateB);
  if (aStdQuGate && bStdQuGate) {
    const auto aScalarGM = aStdQuGate->getScalarGM();
    const auto bScalarGM = bStdQuGate->getScalarGM();
    assert(aScalarGM != nullptr && bScalarGM != nullptr &&
           "Both gate matrices must be ScalarGateMatrix for now");
    const ComplexSquareMatrix& aCMat = aScalarGM->matrix();
    const ComplexSquareMatrix& bCMat = bScalarGM->matrix();
    ComplexSquareMatrix cCMat(1ULL << cnQubits);
    matmulComplexSquareMatrix(aCMat, bCMat, cCMat);
    
    return StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(std::move(cCMat)),
      NoiseChannelPtr(nullptr), // No noise channel for now
      cQubits
    );
  }
  assert(false && "Only implemented matmul for StandardQuantumGate now");
  return nullptr;
}

SuperopQuantumGatePtr StandardQuantumGate::getSuperopGate() const {
  assert(_gateMatrix != nullptr);
  auto scalarGM = getScalarGM();
  assert(scalarGM != nullptr && "Only supporting ScalarGateMatrix for now");
  if (_noiseChannel == nullptr) {
    // If there is no noise channel, the superoperator matrix is just the gate
    // matrix itself.
    auto superopGateGM = std::make_shared<ScalarGateMatrix>(scalarGM->matrix());
    return SuperopQuantumGate::Create(superopGateGM, qubits());
  }

  assert(_noiseChannel->reps.krausRep != nullptr &&
        "We must have KrausRep to compute superoperator matrix");
  const auto& krausOps = _noiseChannel->reps.krausRep->getOps();
  // If the gate matrix and every Kraus operator are N * N, then the 
  // superoperator matrix is (N ** 2) * (N ** 2).
  const size_t N = 1ULL << nQubits();
  auto superopGateGM = std::make_shared<ScalarGateMatrix>(nQubits() * 2);
  auto& superOpM = superopGateGM->matrix();
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
      superOpM.real(row, col) +=
        newKrausOpM.real(r, c) * newKrausOpM.real(rr, cc) +
        newKrausOpM.imag(r, c) * newKrausOpM.imag(rr, cc);
      superOpM.imag(row, col) +=
        newKrausOpM.real(r, c) * newKrausOpM.imag(rr, cc) -
        newKrausOpM.imag(r, c) * newKrausOpM.real(rr, cc);
    } } } }
  }
  return SuperopQuantumGate::Create(superopGateGM, qubits());
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

  if (_noiseChannel == nullptr) {
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
  assert(_superopMatrix != nullptr && "Superop matrix is null");
  double count = static_cast<double>(
    countNonZeroElems(*_superopMatrix, zeroTol));
  
  // superop matrices are treated as 2n-qubit gates
  return count * std::pow<double>(2.0, 1 - 2 * nQubits());
}

/**** Inverse ****/

QuantumGatePtr StandardQuantumGate::inverse() const {
  if (_noiseChannel != nullptr)
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
    qubits()
  );
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


