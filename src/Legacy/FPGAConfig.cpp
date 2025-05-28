#include "cast/Legacy/FPGAConfig.h"

using namespace cast::legacy::fpga;

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

const FPGAGateCategoryTolerance FPGAGateCategoryTolerance::Default{
    .upTol = 1e-8,
    .ncTol = 1e-8,
    .reOnlyTol = 1e-8,
};

const FPGAGateCategoryTolerance FPGAGateCategoryTolerance::Zero{
    .upTol = 0.0, .ncTol = 0.0, .reOnlyTol = 0.0};

const FPGAFusionConfig FPGAFusionConfig::Default{
    .maxUnitaryPermutationSize = 5,
    .ignoreSingleQubitNonCompGates = true,
    .multiTraverse = true,
    .tolerances =
        {
            .upTol = 1e-8,
            .ncTol = 1e-8,
            .reOnlyTol = 1e-8,
        },
};