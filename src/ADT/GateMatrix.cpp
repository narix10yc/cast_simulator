#include "cast/ADT/GateMatrix.h"
#include "llvm/Support/Casting.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

using namespace cast;

std::ostream& ScalarGateMatrix::displayInfo(std::ostream& os, int verbose) const {
  os << "ScalarGateMatrix @ " << this << "\n";
  os << "- nQubits: " << _nQubits << "\n";
  return os;
} // namespace cast

static unsigned permIdx(unsigned idx, const std::vector<int>& flags) {
  unsigned newIdx = 0;
  unsigned k = flags.size();
  for (unsigned b = 0; b < k; b++) {
    newIdx += ((idx & (1U << b)) >> b) << flags[b];
  }
  return newIdx;
}

static ScalarGateMatrixPtr _permute(
    ScalarGateMatrix* sMat, const std::vector<int>& flags) {
  assert(sMat != nullptr);
  assert(utils::isPermutation(flags));
  assert(sMat->nQubits() == flags.size());


  auto matrix = std::make_shared<ScalarGateMatrix>(sMat->nQubits());
  unsigned edgeSize = matrix->matrix().edgeSize();
  for (unsigned r = 0; r < edgeSize; ++r) {
    for (unsigned c = 0; c < edgeSize; ++c) {
      matrix->matrix().real(permIdx(r, flags), permIdx(c, flags)) =
        sMat->matrix().real(r, c);
      matrix->matrix().imag(permIdx(r, flags), permIdx(c, flags)) =
        sMat->matrix().imag(r, c);
    }
  }

  return matrix;
}

static UnitaryPermGateMatrixPtr _permute(
    UnitaryPermGateMatrix* uMat, const std::vector<int>& flags) {
  assert(false && "Not implemented");
  return nullptr;
}

static ParametrizedGateMatrixPtr _permute(
    ParametrizedGateMatrix* pMat, const std::vector<int>& flags) {
  assert(false && "Not implemented");
  return nullptr;
}

GateMatrixPtr cast::permute(GateMatrixPtr gm, const std::vector<int>& flags) {
  if (gm == nullptr)
    return nullptr;

  assert(gm->nQubits() == flags.size());
  bool isIdentity = true;
  for (unsigned i = 0; i < flags.size(); i++) {
    if (flags[i] != i) {
      isIdentity = false;
      break;
    }
  }
  if (isIdentity)
    return gm;

  if (auto* sMat = llvm::dyn_cast<ScalarGateMatrix>(gm.get()))
    return _permute(sMat, flags);
  if (auto* uMat = llvm::dyn_cast<UnitaryPermGateMatrix>(gm.get()))
    return _permute(uMat, flags);
  if (auto* pMat = llvm::dyn_cast<ParametrizedGateMatrix>(gm.get()))
    return _permute(pMat, flags);
  assert(false && "Unknown GateMatrix type");
  return nullptr;
}
