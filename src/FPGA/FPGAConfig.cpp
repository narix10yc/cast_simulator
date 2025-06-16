#include "cast/FPGA/FPGAGateCategory.h"
#include "cast/FPGA/FPGAFusion.h"

using namespace cast::fpga;

const FPGAGateCategory FPGAGateCategory::General(
    static_cast<unsigned>(FPGAGateCategory::fpgaGeneral));

const FPGAGateCategory FPGAGateCategory::SingleQubit(
    static_cast<unsigned>(FPGAGateCategory::fpgaSingleQubit));

const FPGAGateCategory FPGAGateCategory::UnitaryPerm(
    static_cast<unsigned>(FPGAGateCategory::fpgaUnitaryPerm));

const FPGAGateCategory FPGAGateCategory::NonComp(
    static_cast<unsigned>(FPGAGateCategory::fpgaNonComp));

const FPGAGateCategory FPGAGateCategory::RealOnly(
    static_cast<unsigned>(FPGAGateCategory::fpgaRealOnly));

const FPGAFusionConfig FPGAFusionConfig::Default {
  .maxUnitaryPermutationSize = 5,
  .ignoreSingleQubitNonCompGates = true,
  .multiTraverse = true,
  .tolerance = 1e-8
};