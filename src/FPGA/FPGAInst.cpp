#include "cast/FPGA/FPGAInst.h"
#include "cast/Core/QuantumGate.h"

#include "llvm/Support/Casting.h"

using namespace cast;

namespace {
// Normalize angle to (-pi, pi]
double normalizePhase(double phase) {
  constexpr double TWO_PI = 2.0 * M_PI;
  phase = std::fmod(phase - M_PI, TWO_PI);  // shift and wrap
  if (phase < 0)
    phase += TWO_PI;                        // ensure result is in [0, 2π)
  return phase - M_PI;                      // shift back to (-π, π]
}
} // namespace

fpga::FPGAGateCategory cast::fpga::getFPGAGateCategory(
    const cast::QuantumGate* gate, double tol) {
      
  assert(gate != nullptr);
  const auto* stdQuGate = llvm::dyn_cast<StandardQuantumGate>(gate);
  assert(stdQuGate && "FPGA only supports StandardQuantumGate");
  assert(stdQuGate->noiseChannel() == nullptr &&
         "FPGA does not support noise channels");

  auto gm = stdQuGate->gateMatrix();
  FPGAGateCategory cate = FPGAGateCategory::General;

  if (gate->nQubits() == 1)
    cate |= FPGAGateCategory::SingleQubit;

  // handle unitary permutation gates
  if (auto upGM = UnitaryPermGateMatrix::FromGateMatrix(gm.get(), tol)) {
    cate |= FPGAGateCategory::UnitaryPerm;
    
    // Check for non-computational-ness
    bool isNonComp = true;
    for (unsigned i = 0; i < (1U << gate->nQubits()); ++i) {
      const auto phase = upGM->data()[i].phase;
      if (!(std::abs(phase) < tol || 
            std::abs(phase - M_PI) < tol ||
            std::abs(phase + M_PI) < tol ||
            std::abs(phase - M_PI_2) < tol ||
            std::abs(phase + M_PI_2) < tol)) {
        isNonComp = false;
        break;
      }
    }
    if (isNonComp)
      cate |= FPGAGateCategory::NonComp;

    // Check if the gate is real only
    bool isReal = true;
    for (unsigned i = 0; i < (1U << gate->nQubits()); ++i) {
      const auto phase = upGM->data()[i].phase;
      if (!(std::abs(phase) < tol || 
            std::abs(phase - M_PI) < tol ||
            std::abs(phase + M_PI) < tol)) {
        isReal = false;
        break;
      }
    }
    if (isReal) {
      cate |= FPGAGateCategory::RealOnly;
    }
    return cate;
  }

  // handle scalar gates
  const auto* scalarGM = llvm::dyn_cast<ScalarGateMatrix>(gm.get());
  assert(scalarGM && "getFPGAGateCategory only supports scalar gates "
                     "or unitary permutation gates");

  // Check if the scalar gate is real only
  bool isReal = true;
  for (unsigned r = 0; r < scalarGM->matrix().edgeSize(); ++r) {
    for (unsigned c = 0; c < scalarGM->matrix().edgeSize(); ++c) {
      const auto re = scalarGM->matrix().real(r, c);
      const auto im = scalarGM->matrix().imag(r, c);
      if (std::abs(im) > tol){
        isReal = false;
        break;
      }
    }
    if (!isReal)
      break;
  }
  if (isReal) {
    cate |= FPGAGateCategory::RealOnly;
  }

  return cate;
}
