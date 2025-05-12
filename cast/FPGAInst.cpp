#include "cast/FPGAInst.h"
#include "cast/LegacyQuantumGate.h"

using namespace cast;
using namespace cast::impl;
using namespace cast::fpga;

namespace {
inline bool isRealOnlyGate(const LegacyQuantumGate& gate, double reTol) {
  const auto* cMat = gate.gateMatrix.getConstantMatrix();
  assert(cMat);
  for (const auto& cplx : *cMat) {
    if (std::abs(cplx.imag()) > reTol)
      return false;
  }
  return true;
}
} // namespace

FPGAGateCategory
cast::fpga::getFPGAGateCategory(
    const LegacyQuantumGate& gate, const FPGAGateCategoryTolerance &tolerances) {
  switch (gate.gateMatrix.gateKind) {
  case gX:
    return FPGAGateCategory::SingleQubit | FPGAGateCategory::NonComp |
           FPGAGateCategory::RealOnly;
  case gY:
    return FPGAGateCategory::SingleQubit | FPGAGateCategory::NonComp;
  case gZ:
    return FPGAGateCategory::SingleQubit | FPGAGateCategory::NonComp;
  case gP:
    return FPGAGateCategory::SingleQubit | FPGAGateCategory::UnitaryPerm;
  case gH:
    return FPGAGateCategory::SingleQubit;
  case gCX:
    return FPGAGateCategory::NonComp;
  case gCZ:
    return FPGAGateCategory::NonComp;
  case gCP:
    return FPGAGateCategory::UnitaryPerm;
  default:
    break;
  }

  FPGAGateCategory cate = FPGAGateCategory::General;

  if (gate.qubits.size() == 1)
    cate |= FPGAGateCategory::SingleQubit;

  if (const auto* p = gate.gateMatrix.getUnitaryPermMatrix(tolerances.upTol)) {
    bool nonCompFlag = true;
    for (const auto& entry : *p) {
      auto normedPhase = entry.normedPhase();
      if (std::abs(normedPhase) > tolerances.ncTol &&
          std::abs(normedPhase - M_PI_2) > tolerances.ncTol &&
          std::abs(normedPhase + M_PI_2) > tolerances.ncTol &&
          std::abs(normedPhase - M_PI) > tolerances.ncTol &&
          std::abs(normedPhase + M_PI) > tolerances.ncTol) {
        nonCompFlag = false;
        break;
      }
    }

    if (nonCompFlag)
      cate |= FPGAGateCategory::NonComp;
    else
      cate = FPGAGateCategory::UnitaryPerm;
  }

  if (isRealOnlyGate(gate, tolerances.reOnlyTol))
    cate |= FPGAGateCategory::RealOnly;

  return cate;
}
