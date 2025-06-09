#include "cast/FPGA/FPGAInst.h"
#include "cast/Core/QuantumGate.h"

#include "llvm/Support/Casting.h"

using namespace cast;

namespace {
inline bool isRealOnlyGate(const QuantumGate* gate, double reTol) {
  const auto* cMat = gate.gateMatrix.getConstantMatrix();
  assert(cMat);
  for (const auto& cplx : *cMat) {
    if (std::abs(cplx.imag()) > reTol)
      return false;
  }
  return true;
}
} // namespace

fpga::FPGAGateCategory cast::fpga::getFPGAGateCategory(
    const cast::QuantumGate* gate,
    const fpga::FPGAGateCategoryTolerance& tolerances) {
      
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
