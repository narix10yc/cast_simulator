#ifndef CAST_LEGACY_GATE_MATRIX_H
#define CAST_LEGACY_GATE_MATRIX_H

#include "cast/UnitaryPermMatrix.h"
#include "cast/ScalarKind.h"
#include "utils/square_matrix.h"
#include "utils/PODVariant.h"
#include "cast/Polynomial.h"

namespace cast {

  namespace impl {
    enum GateKind : int {
      gUndef = 0,

      // single-qubit gates
      gX = -10,
      gY = -11,
      gZ = -12,
      gH = -13,
      gP = -14,
      gRX = -15,
      gRY = -16,
      gRZ = -17,
      gU = -18,

      // two-qubit gates
      gCX = -100,
      gCNOT = -100,
      gCZ = -101,
      gSWAP = -102,
      gCP = -103,

      // general multi-qubit dense gates
      gU1q = 1,
      gU2q = 2,
      gU3q = 3,
      gU4q = 4,
      gU5q = 5,
      gU6q = 6,
      gU7q = 7,
      // to be defined by nQubits directly
    };

    GateKind String2GateKind(const std::string& s);
    std::string GateKind2String(GateKind t);
  } // namespace impl


class LegacyGateMatrix {
public:
  // specify gate matrix with (up to three) parameters
  using gate_params_t = std::array<utils::PODVariant<int, double>, 3>;
  // unitary permutation matrix type
  using up_matrix_t = cast::UnitaryPermutationMatrix;
  // constant matrix type
  using c_matrix_t = utils::square_matrix<std::complex<double>>;
  // parametrised matrix type
  using p_matrix_t = utils::square_matrix<cast::Polynomial>;
  // signature matrix type
  using sig_matrix_t = utils::square_matrix<std::complex<ScalarKind>>;

private:
  enum ConvertibleKind : int {
    Unknown = -1,
    Convertible = 1,
    UnConvertible = 0
  };
  struct Cache {
    ConvertibleKind isConvertibleToUpMat;
    up_matrix_t upMat;
    ConvertibleKind isConvertibleToCMat;
    c_matrix_t cMat;
    // gate matrix is always convertible to pMat
    ConvertibleKind isConvertibleToPMat;
    p_matrix_t pMat;

    double sigZeroTol;
    double sigOneTol;
    sig_matrix_t sigMat;

    Cache()
      : isConvertibleToUpMat(Unknown), upMat(), isConvertibleToCMat(Unknown),
        cMat(), isConvertibleToPMat(Unknown), pMat(), sigZeroTol(-1.0),
        sigOneTol(-1.0), sigMat() {}
  };

  mutable Cache cache;

  void computeAndCacheUpMat(double tolerance) const;
  void computeAndCacheCMat() const;
  void computeAndCachePMat() const;
  void computeAndCacheSigMat(double zeroTol, double oneTol) const;

public:
  impl::GateKind gateKind;
  gate_params_t gateParams;

  LegacyGateMatrix() : cache(), gateKind(impl::gUndef), gateParams() {}

  LegacyGateMatrix(impl::GateKind gateKind, const gate_params_t& params = {})
      : cache(), gateKind(gateKind), gateParams(params) {}

  explicit LegacyGateMatrix(const up_matrix_t& upMat);
  explicit LegacyGateMatrix(const c_matrix_t& upMat);
  explicit LegacyGateMatrix(const p_matrix_t& upMat);

  static LegacyGateMatrix FromName(
      const std::string& name, const gate_params_t& params = {});

  void permuteSelf(const llvm::SmallVector<int>& flags);

  // get cached unitary perm matrix object associated with this GateMatrix
  const up_matrix_t* getUnitaryPermMatrix(double tolerance = 0.0) const {
    if (cache.isConvertibleToUpMat == Unknown)
      computeAndCacheUpMat(tolerance);
    if (cache.isConvertibleToUpMat == UnConvertible)
      return nullptr;
    return &cache.upMat;
  }

  up_matrix_t* getUnitaryPermMatrix(double tolerance = 0.0) {
    if (cache.isConvertibleToUpMat == Unknown)
      computeAndCacheUpMat(tolerance);
    if (cache.isConvertibleToUpMat == UnConvertible)
      return nullptr;
    return &cache.upMat;
  }

  // get cached constant matrix object associated with this GateMatrix
  const c_matrix_t* getConstantMatrix() const {
    if (cache.isConvertibleToCMat == Unknown)
      computeAndCacheCMat();
    if (cache.isConvertibleToCMat == UnConvertible)
      return nullptr;
    return &cache.cMat;
  }

  c_matrix_t* getConstantMatrix() {
    if (cache.isConvertibleToCMat == Unknown)
      computeAndCacheCMat();
    if (cache.isConvertibleToCMat == UnConvertible)
      return nullptr;
    return &cache.cMat;
  }

  // get cached parametrized matrix object associated with this GateMatrix
  const p_matrix_t& getParametrizedMatrix() const {
    if (cache.isConvertibleToPMat == Unknown)
      computeAndCachePMat();
    assert(cache.isConvertibleToPMat == Convertible);
    return cache.pMat;
  }

  p_matrix_t& getParametrizedMatrix() {
    if (cache.isConvertibleToPMat == Unknown)
      computeAndCachePMat();
    assert(cache.isConvertibleToPMat == Convertible);
    return cache.pMat;
  }

  const sig_matrix_t& getSignatureMatrix(double zeroTol, double oneTol) const {
    if (cache.sigMat.edgeSize() == 0) {
      assert(cache.sigZeroTol < 0.0);
      assert(cache.sigOneTol < 0.0);
      computeAndCacheSigMat(zeroTol, oneTol);
    }
    return cache.sigMat;
  }

  bool isConvertibleToUnitaryPermMatrix(double tolerance) const {
    return getUnitaryPermMatrix(tolerance) != nullptr;
  }

  bool isConvertibleToConstantMatrix() const {
    return getConstantMatrix() != nullptr;
  }

  // @brief Get number of qubits
  int nQubits() const;

  /// @brief Return other * this. That is, A.lmatmul(B) will return BA.
  /// Currently only support cMat
  LegacyGateMatrix lmatmul(const LegacyGateMatrix& other) const;

  std::ostream& printCMat(std::ostream& os) const;

  std::ostream& printPMat(std::ostream& os) const;

  // preset unitary permutation matrices
  static const up_matrix_t MatrixI1_up;
  static const up_matrix_t MatrixI2_up;

  static const up_matrix_t MatrixX_up;
  static const up_matrix_t MatrixY_up;
  static const up_matrix_t MatrixZ_up;

  static const up_matrix_t MatrixCX_up;
  static const up_matrix_t MatrixCZ_up;

  // preset constant matrices
  static const c_matrix_t MatrixI1_c;
  static const c_matrix_t MatrixI2_c;

  static const c_matrix_t MatrixX_c;
  static const c_matrix_t MatrixY_c;
  static const c_matrix_t MatrixZ_c;
  static const c_matrix_t MatrixH_c;

  static const c_matrix_t MatrixCX_c;
  static const c_matrix_t MatrixCZ_c;

  // preset parametrized matrices
  static const p_matrix_t MatrixI1_p;
  static const p_matrix_t MatrixI2_p;

  static const p_matrix_t MatrixX_p;
  static const p_matrix_t MatrixY_p;
  static const p_matrix_t MatrixZ_p;
  static const p_matrix_t MatrixH_p;

  static const p_matrix_t MatrixCX_p;
  static const p_matrix_t MatrixCZ_p;

  // preset gate matrices
  static LegacyGateMatrix H() { return LegacyGateMatrix(MatrixH_c); }
  static LegacyGateMatrix X() { return LegacyGateMatrix(MatrixX_c); }
  static LegacyGateMatrix Y() { return LegacyGateMatrix(MatrixY_c); }
  static LegacyGateMatrix Z() { return LegacyGateMatrix(MatrixZ_c); }
  static LegacyGateMatrix I1() { return LegacyGateMatrix(MatrixI1_c); }
  static LegacyGateMatrix I2() { return LegacyGateMatrix(MatrixI2_c); }
  static LegacyGateMatrix CX() { return LegacyGateMatrix(MatrixCX_c); }
  static LegacyGateMatrix CZ() { return LegacyGateMatrix(MatrixCZ_c); }
};

// An easy name for gate matrix pointer type
using gmp_t = std::shared_ptr<LegacyGateMatrix>;

} // namespace cast

#endif // CAST_LEGACY_GATE_MATRIX_H
