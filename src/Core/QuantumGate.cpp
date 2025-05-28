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
    int q, aIdx, bIdx;
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
      targetQubitsInfo.emplace_back(bQubit, -1, bIdx++);
    }
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


  const auto matmulComplexSquareMatrix = [&](
      const ComplexSquareMatrix& matA,
      const ComplexSquareMatrix& matB,
      ComplexSquareMatrix& matC) -> void {
    for (uint64_t cIdx = 0ULL; cIdx < (1ULL << (2 * cnQubits)); ++cIdx) {
      uint64_t aIdxBegin = utils::pext64(cIdx, aPextMask) & aZeroingMask;
      uint64_t bIdxBegin = utils::pext64(cIdx, bPextMask) & bZeroingMask;

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

  const auto* standardGateA = llvm::dyn_cast<StandardQuantumGate>(gateA);
  const auto* standardGateB = llvm::dyn_cast<StandardQuantumGate>(gateB);
  if (standardGateA && standardGateB) {
    const auto* aScalarMatPtr =
      llvm::dyn_cast<ScalarGateMatrix>(standardGateA->gateMatrix().get());
    const auto* bScalarMatPtr =
      llvm::dyn_cast<ScalarGateMatrix>(standardGateB->gateMatrix().get());
    assert(aScalarMatPtr != nullptr && bScalarMatPtr != nullptr &&
          "Both gate matrices must be ScalarGateMatrix for now");
    const ComplexSquareMatrix& aCMat = aScalarMatPtr->matrix();
    const ComplexSquareMatrix& bCMat = bScalarMatPtr->matrix();
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