#ifndef CAST_FPGA_FPGACONFIG_H
#define CAST_FPGA_FPGACONFIG_H

namespace cast::fpga {

class FPGAGateCategory {
public:
  enum Kind : unsigned {
    fpgaGeneral = 0,
    fpgaSingleQubit = 0b0001,

    // unitary permutation
    fpgaUnitaryPerm = 0b0010,

    // Non-computational is a special subclass of unitary permutation where all
    // non-zero entries are +1, -1, +i, -i.
    fpgaNonComp = 0b0110,
    fpgaRealOnly = 0b1000,
  };

  unsigned category;

  explicit FPGAGateCategory(unsigned category) : category(category) {}

  bool is(Kind kind) const {
    return (category & static_cast<unsigned>(kind)) ==
           static_cast<unsigned>(kind);
  }

  bool isNot(Kind kind) const { return !is(kind); }

  FPGAGateCategory& operator|=(const Kind& kind) {
    category |= static_cast<unsigned>(kind);
    return *this;
  }

  FPGAGateCategory operator|(const Kind& kind) const {
    return FPGAGateCategory(category | static_cast<unsigned>(kind));
  }

  FPGAGateCategory& operator|=(const FPGAGateCategory& other) {
    category |= static_cast<unsigned>(other.category);
    return *this;
  }

  FPGAGateCategory operator|(const FPGAGateCategory& other) const {
    return FPGAGateCategory(category | static_cast<unsigned>(other.category));
  }

  static const FPGAGateCategory General;
  static const FPGAGateCategory SingleQubit;
  static const FPGAGateCategory UnitaryPerm;
  static const FPGAGateCategory NonComp; // NonComp implies UnitaryPerm
  static const FPGAGateCategory RealOnly;
};

struct FPGAGateCategoryTolerance {
  double upTol;     // unitary perm gate tolerance
  double ncTol;     // non-computational gate tolerance
  double reOnlyTol; // real only gate tolerance

  static const FPGAGateCategoryTolerance Default;
  static const FPGAGateCategoryTolerance Zero;
};

struct FPGAInstGenConfig {
public:
  int nLocalQubits;
  int gridSize;

  // If off, apply sequential instruction generation on the default order of
  // blocks present in legacy::CircuitGraph
  bool selectiveGenerationMode;
  int maxUpSize;

  FPGAGateCategoryTolerance tolerances;

  int getNOnChipQubits() const { return nLocalQubits + 2 * gridSize; }
};

struct FPGAFusionConfig {
  int maxUnitaryPermutationSize;
  bool ignoreSingleQubitNonCompGates;
  bool multiTraverse;
  FPGAGateCategoryTolerance tolerances;

  static const FPGAFusionConfig Default;
};

} // namespace cast::fpga

#endif // CAST_FPGA_FPGACONFIG_H