#include "cast/Legacy/QuantumGate.h"
#include "utils/iocolor.h"
#include "utils/utils.h"
#include <cmath>
#include <iomanip>

using namespace cast::legacy;
using namespace IOColor;

std::ostream& QuantumGate::displayInfo(std::ostream& os) const {
  os << CYAN("legacy::QuantumGate Info\n") << "- Target Qubits ";
  utils::printArray(os, llvm::ArrayRef(qubits.begin(), qubits.size())) << "\n";
  os << "- Matrix:\n";
  return gateMatrix.printCMat(os);
}

void QuantumGate::sortQubits() {
  const auto nQubits = qubits.size();
  llvm::SmallVector<int> indices(nQubits);
  for (unsigned i = 0; i < nQubits; i++)
    indices[i] = i;

  std::ranges::sort(indices,[&qubits = this->qubits](int i, int j) {
    return qubits[i] < qubits[j];
  });

  llvm::SmallVector<int> newQubits(nQubits);
  for (unsigned i = 0; i < nQubits; i++)
    newQubits[i] = qubits[indices[i]];

  qubits = std::move(newQubits);
  gateMatrix.permuteSelf({indices.begin(), indices.end()});
}

// helper functions in legacy::QuantumGate::lmatmul
namespace {
// inline legacy::QuantumGate lmatmul_up_up(
//     const legacy::GateMatrix::up_matrix_t& aUp,
//     const legacy::GateMatrix::up_matrix_t& bUp,
//     const llvm::SmallVector<int>& aQubits,
//     const llvm::SmallVector<int>& bQubits) {
//   const int anQubits = aQubits.size();
//   const int bnQubits = bQubits.size();
//
//   llvm::SmallVector<int> cQubits;
//   for (const auto& q : aQubits)
//     cQubits.push_back(q);
//   for (const auto& q : bQubits) {
//     if (std::ranges::find(cQubits, q) == cQubits.end())
//       cQubits.push_back(q);
//   }
//   std::ranges::sort(cQubits);
//
//   const int cnQubits = cQubits.size();
//   int i = 0;
//   uint64_t aMask = 0;
//   uint64_t bMask = 0;
//   for (auto it = cQubits.begin(); it != cQubits.end(); ++it) {
//     if (std::ranges::find(aQubits, *it) != aQubits.end())
//       aMask |= (1 << i);
//     if (std::ranges::find(bQubits, *it) != bQubits.end())
//       bMask |= (1 << i);
//     i++;
//   }
//
//   legacy::GateMatrix::up_matrix_t cUp(1 << cnQubits);
//   for (uint64_t idx = 0; idx < (1 << cnQubits); idx++) {
//     auto bIdx = utils::pext64(idx, bMask, cQubits.back());
//     auto aIdx = utils::pext64(idx, aMask, cQubits.back());
//     auto phase = bUp[bIdx].phase + aUp[aIdx].phase;
//     cUp[idx] = {bUp[bIdx].index, phase};
//   }
//   return legacy::QuantumGate(legacy::GateMatrix(cUp), cQubits);
// }
} // anonymous namespace

QuantumGate QuantumGate::lmatmul(const QuantumGate& other) const {
  // C = B.lmatmul(A) returns matrix multiplication C = AB
  // A is other, B is this
  const auto& aQubits = other.qubits;
  const auto& bQubits = qubits;
  const int anQubits = aQubits.size();
  const int bnQubits = bQubits.size();

  struct TargetQubitsInfo {
    int q, aIdx, bIdx;
  };

  // setup result gate target qubits
  llvm::SmallVector<TargetQubitsInfo> targetQubitsInfo;
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
  llvm::SmallVector<uint64_t> aSharedQubitShifts, bSharedQubitShifts;
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
  llvm::SmallVector<int> cQubits;
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

  // unitary perm gate matrix
  // {
  // auto aUpMat = gateMatrix.getUnitaryPermMatrix();
  // auto bUpMat = other.gateMatrix.getUnitaryPermMatrix();
  // if (aUpMat.has_value() && bUpMat.has_value())
  //     return lmatmul_up_up(aUpMat.value(), bUpMat.value(), qubits,
  //     other.qubits);
  // }

  // const matrix
  const GateMatrix::c_matrix_t* aCMat;
  const GateMatrix::c_matrix_t* bCMat;
  if ((aCMat = other.gateMatrix.getConstantMatrix())
      && (bCMat = gateMatrix.getConstantMatrix())) {
    GateMatrix::c_matrix_t cCMat(1 << cnQubits);
    // main loop
    for (uint64_t i = 0ULL; i < (1ULL << (2 * cnQubits)); i++) {
      uint64_t aIdxBegin = utils::pext64(i, aPextMask) & aZeroingMask;
      uint64_t bIdxBegin = utils::pext64(i, bPextMask) & bZeroingMask;

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

      for (uint64_t s = 0; s < (1ULL << contractionWidth); s++) {
        uint64_t aIdx = aIdxBegin;
        uint64_t bIdx = bIdxBegin;
        for (unsigned bit = 0; bit < contractionWidth; bit++) {
          if (s & (1 << bit)) {
            aIdx += aSharedQubitShifts[bit];
            bIdx += bSharedQubitShifts[bit];
          }
        }
        // std::cerr << "  aIdx = " << aIdx << ": " << aCMat->data()[aIdx] << ";"
                  // << "  bIdx = " << bIdx << ": " << bCMat->data()[bIdx] << "\n";
        cCMat[i] += aCMat->data()[aIdx] * bCMat->data()[bIdx];
      }
    }
    return QuantumGate(GateMatrix(cCMat), cQubits);
  }

  // otherwise, parametrised matrix
  auto aPMat = other.gateMatrix.getParametrizedMatrix();
  auto bPMat = gateMatrix.getParametrizedMatrix();
  GateMatrix::p_matrix_t cPMat(1 << cnQubits);
  // main loop
  for (uint64_t i = 0ULL; i < (1ULL << (2 * cnQubits)); i++) {
    uint64_t aIdxBegin = utils::pdep64(i, aPextMask) & aZeroingMask;
    uint64_t bIdxBegin = utils::pdep64(i, bPextMask) & bZeroingMask;

    for (uint64_t s = 0; s < (1ULL << contractionWidth); s++) {
      uint64_t aIdx = aIdxBegin;
      uint64_t bIdx = bIdxBegin;
      for (unsigned bit = 0; bit < contractionWidth; bit++) {
        if (s & (1 << bit)) {
          aIdx += aSharedQubitShifts[bit];
          bIdx += bSharedQubitShifts[bit];
        }
      }
      cPMat[i] += aPMat[aIdx] * bPMat[bIdx];
    }
  }

  return QuantumGate(GateMatrix(cPMat), cQubits);
}

namespace { // legacy::QuantumGate::opCount helper functions
inline int opCount_c(const GateMatrix::c_matrix_t& mat, double zeroTol) {
  int count = 0;
  for (const auto& data : mat) {
    if (std::abs(data.real()) >= zeroTol)
      ++count;
    if (std::abs(data.imag()) >= zeroTol)
      ++count;
  }
  return 2 * count;
}

inline int opCount_p(const GateMatrix::p_matrix_t& mat, double zeroTol) {
  int count = 0;
  for (const auto& data : mat) {
    auto ev = data.getValue();
    if (ev.first) {
      if (std::abs(ev.second.real()) >= zeroTol)
        ++count;
      if (std::abs(ev.second.imag()) >= zeroTol)
        ++count;
    } else
      count += 2;
  }
  return 2 * count;
}

} // anonymous namespace

double QuantumGate::opCount(double zeroTol) const {
  if (opCountCache >= 0)
    return opCountCache;

  if (const auto* cMat = gateMatrix.getConstantMatrix())
    return opCount_c(*cMat, zeroTol) * std::pow(2.0, -nQubits());
  return opCount_p(gateMatrix.getParametrizedMatrix(), zeroTol) * std::pow(2.0, -nQubits());

  assert(false && "opCount only implemented for c and p matrices");

  return -1;
}

// bool legacy::QuantumGate::approximatelegacy::GateMatrix(double thres) {
//     const auto approxCplx = [thres=thres](std::complex<double>& cplx) -> bool
//     {
//         bool flag = false;
//         if (std::abs(cplx.real()) < thres) {
//             cplx.real(0.0);
//             flag = true;
//         }
//         else if (std::abs(cplx.real() - 1.0) < thres) {
//             cplx.real(1.0);
//             flag = true;
//         }
//         else if (std::abs(cplx.real() + 1.0) < thres) {
//             cplx.real(-1.0);
//             flag = true;
//         }

//         if (std::abs(cplx.imag()) < thres) {
//             cplx.imag(0.0);
//             flag = true;
//         }
//         else if (std::abs(cplx.imag() - 1.0) < thres) {
//             cplx.imag(1.0);
//             flag = true;
//         }
//         else if (std::abs(cplx.imag() + 1.0) < thres) {
//             cplx.imag(-1.0);
//             flag = true;
//         }
//         return flag;
//     };

//     auto* p = std::get_if<legacy::GateMatrix::c_matrix_t>(&gateMatrix._matrix);
//     if (p == nullptr)
//         return false;
//     bool flag = false;
//     for (auto& data : p->data) {
//         if (approxCplx(data))
//             flag = true;
//     }
//     return flag;
// }
