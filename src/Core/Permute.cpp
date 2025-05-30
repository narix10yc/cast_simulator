/*
 * This file implements the two cast::permute functions for GateMatrix and
 * NoiseChannel.
 */
#include "cast/Core/QuantumGate.h"
#include "llvm/Support/Casting.h"

using namespace cast;

static bool isPermutation(const std::vector<int>& flags) {
  assert(flags.size() > 0 && "Flags must not be empty");
  std::vector<int> copy(flags);
  std::ranges::sort(copy);
  for (unsigned i = 0, S = copy.size(); i < S; ++i) {
    if (copy[i] != i)
      return false;
  }
  return true;
}

static bool isIdentityPermutation(const std::vector<int>& flags) {
  assert(flags.size() > 0 && "Flags must not be empty");
  for (unsigned i = 0, S = flags.size(); i < S; ++i) {
    if (flags[i] != i)
      return false;
  }
  return true;
}

// For each bit b, newIdx[flags[b]] = idx[b].
static unsigned permIdx(unsigned idx, const std::vector<int>& flags) {
  unsigned newIdx = 0;
  unsigned k = flags.size();
  for (unsigned b = 0; b < k; b++) {
    newIdx += ((idx & (1U << b)) >> b) << flags[b];
  }
  return newIdx;
}

static ScalarGateMatrixPtr permuteSGateMatrix(
    const ScalarGateMatrix* sMat, const std::vector<int>& flags) {
  assert(sMat != nullptr);
  assert(isPermutation(flags));
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

static UnitaryPermGateMatrixPtr permuteUPGateMatrix(
    UnitaryPermGateMatrix* uMat, const std::vector<int>& flags) {
  assert(false && "Not implemented");
  return nullptr;
}

static ParametrizedGateMatrixPtr permutePGateMatrix(
    ParametrizedGateMatrix* pMat, const std::vector<int>& flags) {
  assert(false && "Not implemented");
  return nullptr;
}

GateMatrixPtr cast::permute(GateMatrixPtr gm, const std::vector<int>& flags) {
  assert(isPermutation(flags) && "Flags must be a permutation");
  assert(gm->nQubits() == flags.size());

  if (gm == nullptr)
    return nullptr;
  if (isIdentityPermutation(flags))
    return gm;

  if (auto* sMat = llvm::dyn_cast<ScalarGateMatrix>(gm.get()))
    return permuteSGateMatrix(sMat, flags);
  if (auto* uMat = llvm::dyn_cast<UnitaryPermGateMatrix>(gm.get()))
    return permuteUPGateMatrix(uMat, flags);
  if (auto* pMat = llvm::dyn_cast<ParametrizedGateMatrix>(gm.get()))
    return permutePGateMatrix(pMat, flags);
  assert(false && "Unknown GateMatrix type");
  return nullptr;
}

NoiseChannelPtr cast::permute(
    NoiseChannelPtr nc, const std::vector<int>& flags) {
  assert(isPermutation(flags) && "Flags must be a permutation");
  
  if (nc == nullptr)
    return nullptr;
  assert(nc->nQubits() == flags.size());
  if (isIdentityPermutation(flags))
    return nc;

  assert(nc->reps.krausRep && "Only supporting permuting by KrausRep for now");
  auto nQubits = nc->nQubits();
  const auto& krausOps = nc->reps.krausRep->getOps();
  auto newKrausRep = std::make_shared<KrausRep>(nQubits);
  for (const auto& krausOp : krausOps) {
    auto newMatrix = permuteSGateMatrix(&krausOp, flags);
    newKrausRep->addMatrix(std::move(*std::move(newMatrix)));
  }
  return std::make_shared<NoiseChannel>(newKrausRep);
}
