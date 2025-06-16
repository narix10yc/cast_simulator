#ifndef CAST_FPGA_FPGAGATECATEGORY_H
#define CAST_FPGA_FPGAGATECATEGORY_H

namespace cast {
  class QuantumGate;
} // namespace cast

namespace cast::fpga {

class FPGAGateCategory {
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
public:
  FPGAGateCategory(unsigned kind) : category(kind) {}

  operator unsigned() const { return category; }

  bool is(FPGAGateCategory cate) const {
    return (category & static_cast<unsigned>(cate)) ==
           static_cast<unsigned>(cate);
  }

  bool isNot(FPGAGateCategory cate) const { return !is(cate); }

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

/// @brief Get the FPGA gate category for a given quantum gate
/// @param tol: the tolerance value. Will be used to check 3 things:
///   1. If the gate is real only, i.e., all imaginary parts are within tol.
///   2. If the gate is unitary permutation, i.e., all phases are close to
///      0, pi, -pi, pi/2, -pi/2.
///   3. If the gate is a non-computational, i.e., all non-zero elements are 
///      close to 1, -1, i or -i.
/// The exact tolerance varies a little. In particular, for unitary permutation,
/// the tolerance is used to check the phases, not the magnitudes.
FPGAGateCategory getFPGAGateCategory(const cast::QuantumGate* gate,
                                     double tol = 1e-8);

} // namespace cast::fpga

#endif // CAST_FPGA_FPGAGATECATEGORY_H