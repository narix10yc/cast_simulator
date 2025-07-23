#include "cast/Legacy/GateMatrix.h"
#include "utils/iocolor.h"
#include <cmath>
#include <iomanip>

using namespace IOColor;
using namespace cast::legacy;
using Polynomial = cast::Polynomial;
using Monomial = cast::Monomial;

GateKind cast::legacy::String2GateKind(const std::string& s) {
  if (s == "x")
    return gX;
  if (s == "y")
    return gY;
  if (s == "z")
    return gZ;
  if (s == "h")
    return gH;
  if (s == "u3" || s == "u1q")
    return gU;
  if (s == "cx")
    return gCX;
  if (s == "cz")
    return gCZ;
  if (s == "cp")
    return gCP;

  assert(false && "Unimplemented cast::impl::String2GateKind");
  return gUndef;
}

std::string cast::legacy::GateKind2String(GateKind t) {
  switch (t) {
  case gX:
    return "x";
  case gY:
    return "y";
  case gZ:
    return "z";
  case gH:
    return "h";
  case gU:
    return "u1q";

  case gCX:
    return "cx";
  case gCZ:
    return "cz";
  case gCP:
    return "cp";

  default:
    return "u" + std::to_string(t) + "q";
  }
}

using p_matrix_t = GateMatrix::p_matrix_t;
using up_matrix_t = GateMatrix::up_matrix_t;
using c_matrix_t = GateMatrix::c_matrix_t;
using gate_params_t = GateMatrix::gate_params_t;

#pragma region Static UP Matrices
const up_matrix_t GateMatrix::MatrixI1_up{{0, 0.0}, {1, 0.0}};

const up_matrix_t GateMatrix::MatrixI2_up{
    {0, 0.0}, {1, 0.0}, {2, 0.0}, {3, 0.0}};

const up_matrix_t GateMatrix::MatrixX_up{{1, 0.0}, {0, 0.0}};

const up_matrix_t GateMatrix::MatrixY_up{{1, -M_PI_2}, {0, M_PI}};

const up_matrix_t GateMatrix::MatrixZ_up{{0, 0.0}, {1, M_PI}};

const up_matrix_t GateMatrix::MatrixCX_up{
    {0, 0.0}, {3, 0.0}, {2, 0.0}, {1, 0.0}};

const up_matrix_t GateMatrix::MatrixCZ_up{
    {0, 0.0}, {1, 0.0}, {2, 0.0}, {3, M_PI}};

#pragma endregion

#pragma region Static Constant Matrices
const c_matrix_t GateMatrix::MatrixI1_c =
    c_matrix_t{{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};

const c_matrix_t GateMatrix::MatrixI2_c{{1.0, 0.0},
                                        {0.0, 0.0},
                                        {0.0, 0.0},
                                        {0.0, 0.0},
                                        {0.0, 0.0},
                                        {1.0, 0.0},
                                        {0.0, 0.0},
                                        {0.0, 0.0},
                                        {0.0, 0.0},
                                        {0.0, 0.0},
                                        {1.0, 0.0},
                                        {0.0, 0.0},
                                        {0.0, 0.0},
                                        {0.0, 0.0},
                                        {0.0, 0.0},
                                        {1.0, 0.0}};

const c_matrix_t GateMatrix::MatrixX_c{
    {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

const c_matrix_t GateMatrix::MatrixY_c{
    {0.0, 0.0}, {0.0, -1.0}, {0.0, 1.0}, {0.0, 0.0}};

const c_matrix_t GateMatrix::MatrixZ_c{
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}};

const c_matrix_t GateMatrix::MatrixH_c = c_matrix_t(
    {{M_SQRT1_2, 0.0}, {M_SQRT1_2, 0.0}, {M_SQRT1_2, 0.0}, {-M_SQRT1_2, 0.0}});

const c_matrix_t GateMatrix::MatrixCX_c{
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
};

const c_matrix_t GateMatrix::MatrixCZ_c{
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {-1.0, 0.0},
};

#pragma endregion

#pragma region Static Parametrized Matrices
const p_matrix_t GateMatrix::MatrixI1_p = {
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};

const p_matrix_t GateMatrix::MatrixI2_p = {{1.0, 0.0},
                                           {0.0, 0.0},
                                           {0.0, 0.0},
                                           {0.0, 0.0},
                                           {0.0, 0.0},
                                           {1.0, 0.0},
                                           {0.0, 0.0},
                                           {0.0, 0.0},
                                           {0.0, 0.0},
                                           {0.0, 0.0},
                                           {1.0, 0.0},
                                           {0.0, 0.0},
                                           {0.0, 0.0},
                                           {0.0, 0.0},
                                           {0.0, 0.0},
                                           {1.0, 0.0}};

const p_matrix_t GateMatrix::MatrixX_p =
    p_matrix_t({{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}});

const p_matrix_t
    GateMatrix::MatrixY_p({{0.0, 0.0}, {0.0, -1.0}, {0.0, 1.0}, {0.0, 0.0}});

const p_matrix_t GateMatrix::MatrixZ_p =
    p_matrix_t({{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}});

const p_matrix_t GateMatrix::MatrixH_p = p_matrix_t(
    {{M_SQRT1_2, 0.0}, {M_SQRT1_2, 0.0}, {M_SQRT1_2, 0.0}, {-M_SQRT1_2, 0.0}});

const p_matrix_t GateMatrix::MatrixCX_p{
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
};

const p_matrix_t GateMatrix::MatrixCZ_p = p_matrix_t({
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {1.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {0.0, 0.0},
    {-1.0, 0.0},
});

#pragma endregion

int getNumActiveParams(const gate_params_t& params) {
  unsigned s = params.size();
  for (unsigned i = 0; i < s; i++) {
    if (!params[i].holdingValue())
      return i;
  }
  return s;
}

GateMatrix::GateMatrix(const up_matrix_t& upMat) : cache(), gateParams() {
  auto size = upMat.edgeSize();
  gateKind = static_cast<GateKind>(static_cast<int>(std::log2(size)));
  assert(1 << static_cast<int>(gateKind) == size);

  cache.upMat = upMat;
  cache.isConvertibleToUpMat = Convertible;
}

GateMatrix::GateMatrix(const c_matrix_t& cMat) : cache(), gateParams() {
  auto size = cMat.edgeSize();
  gateKind = static_cast<GateKind>(static_cast<int>(std::log2(size)));
  assert(1 << static_cast<int>(gateKind) == size);

  cache.cMat = cMat;
  cache.isConvertibleToCMat = Convertible;
}

GateMatrix::GateMatrix(const p_matrix_t& pMat) : cache(), gateParams() {
  auto size = pMat.edgeSize();
  gateKind = static_cast<GateKind>(static_cast<int>(std::log2(size)));
  assert(1 << static_cast<int>(gateKind) == size);

  cache.pMat = pMat;
  cache.isConvertibleToPMat = Convertible;
}

GateMatrix GateMatrix::FromName(const std::string& name,
                                const gate_params_t& params) {
  if (name == "x") {
    assert(getNumActiveParams(params) == 0 && "X gate has 0 parameter");
    return GateMatrix(gX);
  }
  if (name == "y") {
    assert(getNumActiveParams(params) == 0 && "Y gate has 0 parameter");
    return GateMatrix(gY);
  }
  if (name == "z") {
    assert(getNumActiveParams(params) == 0 && "Z gate has 0 parameter");
    return GateMatrix(gZ);
  }
  if (name == "h") {
    assert(getNumActiveParams(params) == 0 && "H gate has 0 parameter");
    return GateMatrix(gH);
  }
  if (name == "p") {
    assert(getNumActiveParams(params) == 1 && "P gate has 1 parameter");
    return GateMatrix(gP, params);
  }
  if (name == "rx") {
    assert(getNumActiveParams(params) == 1 && "RX gate has 1 parameter");
    return GateMatrix(gRX, params);
  }
  if (name == "ry") {
    assert(getNumActiveParams(params) == 1 && "RY gate has 1 parameter");
    return GateMatrix(gRY, params);
  }
  if (name == "rz") {
    assert(getNumActiveParams(params) == 1 && "RZ gate has 1 parameter");
    return GateMatrix(gRZ, params);
  }
  if (name == "u3" || name == "u1q") {
    assert(getNumActiveParams(params) == 3 && "U3 (U1q) gate has 3 parameters");
    return GateMatrix(gU, params);
  }

  if (name == "cx") {
    assert(getNumActiveParams(params) == 0 && "CX gate has 0 parameter");
    return GateMatrix(gCX);
  }
  if (name == "cz") {
    assert(getNumActiveParams(params) == 0 && "CZ gate has 0 parameter");
    return GateMatrix(gCZ);
  }
  if (name == "cp") {
    assert(getNumActiveParams(params) == 1 && "CP gate has 1 parameter");
    return GateMatrix(gCP, params);
  }

  assert(false && "Unsupported gate");
  return GateMatrix(gUndef);
}

std::ostream& GateMatrix::printCMat(std::ostream& os) const {
  const auto* cMat = getConstantMatrix();
  assert(cMat != nullptr && "GateMatrix is not convertible to cmatrix_t");
  return utils::printComplexMatrixF64(os, *cMat);
}

std::ostream& GateMatrix::printPMat(std::ostream& os) const {
  const auto& pMat = getParametrizedMatrix();
  auto edgeSize = pMat.edgeSize();
  for (size_t r = 0; r < edgeSize; r++) {
    for (size_t c = 0; c < edgeSize; c++) {
      os << "[" << r << "," << c << "]: ";
      pMat(r, c).print(os) << "\n";
    }
  }
  return os;
}

void GateMatrix::permuteSelf(const llvm::SmallVector<int>& flags) {
  // single qubit gates
  if (nQubits() == 1) {
    assert(flags.size() == 1);
    return;
  }
  switch (gateKind) {
  // two-qubit symmetric gates
  case gCX:
    assert(flags.size() == 2);
    return;
  case gCZ:
    assert(flags.size() == 2);
    return;
  case gSWAP:
    assert(flags.size() == 2);
    return;
  case gCP:
    assert(flags.size() == 2);
    return;

  default:
    break;
  }

  // TODO: This is not efficient -- we may permute multiple matrices
  assert(gateKind >= 1);
  if (cache.isConvertibleToUpMat == Convertible)
    cache.upMat = cache.upMat.permute(flags);
  if (cache.isConvertibleToCMat == Convertible)
    cache.cMat = cache.cMat.permute(flags);
  if (cache.isConvertibleToPMat == Convertible)
    cache.pMat = cache.pMat.permute(flags);
}

int GateMatrix::nQubits() const {
  switch (gateKind) {
  case gX:
    return 1;
  case gY:
    return 1;
  case gZ:
    return 1;
  case gH:
    return 1;
  case gP:
    return 1;
  case gRX:
    return 1;
  case gRY:
    return 1;
  case gRZ:
    return 1;
  case gU:
    return 1;

  case gCX:
    return 2;
  case gCZ:
    return 2;
  case gSWAP:
    return 2;
  case gCP:
    return 2;

  default:
    assert(gateKind >= 1);
    return static_cast<int>(gateKind);
  }
}

namespace { // matrix conversion

inline p_matrix_t matCvt_gp_to_p(GateKind kind, const gate_params_t& params) {
  switch (kind) {
  case gX:
    return GateMatrix::MatrixX_p;
  case gY:
    return GateMatrix::MatrixY_p;
  case gZ:
    return GateMatrix::MatrixZ_p;
  case gH:
    return GateMatrix::MatrixH_p;
  case gP: {
    assert(params[0].holdingValue());
    if (params[0].is<int>())
      return p_matrix_t{
          {1.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {std::cos(params[0].get<int>()), std::sin(params[0].get<int>())}};
    return p_matrix_t{
        {1.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {Monomial::Expi(params[0].get<double>())} // expi(theta)
    };
  }

  case gU: {
    assert(params[0].holdingValue());
    assert(params[1].holdingValue());
    assert(params[2].holdingValue());

    p_matrix_t pmat{{1.0, 0.0}, {-1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
    // theta
    if (params[0].is<double>()) {
      const auto phase = params[0].get<double>();
      pmat(0, 0) *= std::complex<double>(std::cos(phase), 0.0);
      pmat(0, 1) *= std::complex<double>(std::sin(phase), 0.0);
      pmat(1, 0) *= std::complex<double>(std::sin(phase), 0.0);
      pmat(1, 1) *= std::complex<double>(std::cos(phase), 0.0);
    } else {
      const auto ref = params[0].get<int>();
      pmat(0, 0) *= Monomial::Cosine(ref);
      pmat(0, 1) *= Monomial::Sine(ref);
      pmat(1, 0) *= Monomial::Sine(ref);
      pmat(1, 1) *= Monomial::Cosine(ref);
    }
    // phi
    if (params[1].is<double>()) {
      const auto phase = params[1].get<double>();
      pmat(1, 0) *= std::complex<double>(std::cos(phase), std::sin(phase));
      pmat(1, 1) *= std::complex<double>(std::cos(phase), std::sin(phase));
    } else {
      const auto ref = params[1].get<int>();
      pmat(1, 0) *= Monomial::Expi(ref);
      pmat(1, 1) *= Monomial::Expi(ref);
    }
    // lambda
    if (params[2].is<double>()) {
      const auto phase = params[2].get<double>();
      pmat(0, 1) *= std::complex<double>(std::cos(phase), std::sin(phase));
      pmat(1, 1) *= std::complex<double>(std::cos(phase), std::sin(phase));
    } else {
      const auto ref = params[2].get<int>();
      pmat(0, 1) *= Monomial::Expi(ref);
      pmat(1, 1) *= Monomial::Expi(ref);
    }
    return pmat;
  }

  case gCX:
    return GateMatrix::MatrixCX_p;
  case gCZ:
    return GateMatrix::MatrixCZ_p;

  default:
    assert(false && "Unsupported cvtMat_gp_to_p yet");
    return {};
  }
}

inline c_matrix_t matCvt_up_to_c(const up_matrix_t& up) {
  const auto s = up.edgeSize();
  c_matrix_t cmat(s);
  for (unsigned i = 0; i < s; i++)
    cmat[up[i].index] = {std::cos(up[i].phase), std::sin(up[i].phase)};

  return cmat;
}

} // namespace

// Two paths lead to upMat: gpMat or cMat
void GateMatrix::computeAndCacheUpMat(double tolerance) const {
  assert(cache.isConvertibleToUpMat == Unknown);
  const auto setConvertible = [&](const up_matrix_t& upMat) {
    cache.upMat = upMat;
    cache.isConvertibleToUpMat = Convertible;
  };

  switch (gateKind) {
  case gX: {
    setConvertible(MatrixX_up);
    return;
  }
  case gY: {
    setConvertible(MatrixY_up);
    return;
  }
  case gZ: {
    setConvertible(MatrixZ_up);
    return;
  }
  case gP: {
    if (gateParams[0].isNot<double>()) {
      cache.isConvertibleToUpMat = UnConvertible;
      return;
    }
    setConvertible({{0, 0.0}, {1, gateParams[1].get<double>()}});
    return;
  }
  case gCX: {
    setConvertible(MatrixCX_up);
    return;
  }
  case gCZ: {
    setConvertible(MatrixCZ_up);
    return;
  }
  case gCP: {
    if (gateParams[0].isNot<double>()) {
      cache.isConvertibleToUpMat = UnConvertible;
      return;
    }
    setConvertible(
        {{0, 0.0}, {0, 0.0}, {0, 0.0}, {1, gateParams[0].get<double>()}});
    return;
  }
  default:
    break;
  }

  // check if convertible to upMat from cMat
  const auto* cMat = getConstantMatrix();
  if (cMat == nullptr) {
    cache.isConvertibleToUpMat = UnConvertible;
    return;
  }
  const auto edgeSize = cMat->edgeSize();
  cache.upMat = up_matrix_t(edgeSize);
  for (size_t r = 0; r < edgeSize; r++) {
    bool rowFlag = false;
    for (size_t c = 0; c < edgeSize; c++) {
      const auto& cplx = cMat->rc(r, c);
      if (std::abs(cplx) > tolerance) {
        if (rowFlag) {
          cache.isConvertibleToUpMat = UnConvertible;
          return;
        }
        rowFlag = true;
        cache.upMat[r] = {c, std::atan2(cplx.imag(), cplx.real())};
      }
    }
    if (!rowFlag) {
      cache.isConvertibleToUpMat = UnConvertible;
      return;
    }
  }
  cache.isConvertibleToUpMat = Convertible;
  return;
}

void GateMatrix::computeAndCacheCMat() const {
  assert(cache.isConvertibleToUpMat == Unknown);
  // try convert from upMat
  if (cache.isConvertibleToUpMat == Convertible) {
    const auto edgeSize = cache.upMat.edgeSize();
    cache.cMat = c_matrix_t(edgeSize);
    for (unsigned i = 0; i < edgeSize; ++i) {
      const auto& idx = cache.upMat[i].index;
      const auto& phase = cache.upMat[i].phase;
      cache.cMat[idx] = {std::cos(phase), std::sin(phase)};
    }
    cache.isConvertibleToCMat = Convertible;
    return;
  }

  const auto setConvertible = [&](const c_matrix_t& cMat) {
    cache.cMat = cMat;
    cache.isConvertibleToCMat = Convertible;
  };

  // try convert from gpMat
  switch (gateKind) {
  case gX: {
    setConvertible(MatrixX_c);
    return;
  }
  case gY: {
    setConvertible(MatrixY_c);
    return;
  }
  case gZ: {
    setConvertible(MatrixZ_c);
    return;
  }
  case gH: {
    setConvertible(MatrixH_c);
    return;
  }
  case gP: {
    if (gateParams[0].isNot<double>()) {
      cache.isConvertibleToCMat = UnConvertible;
      return;
    }
    const auto phase = gateParams[0].get<double>();
    setConvertible({{1.0, 0.0},
                    {0.0, 0.0},
                    {0.0, 0.0},
                    {std::cos(phase), std::sin(phase)}});
    return;
  }

  case gU: {
    if (gateParams[0].is<double>() && gateParams[1].is<double>() &&
        gateParams[2].is<double>()) {
      const auto pTheta = gateParams[0].get<double>();
      const auto pPhi = gateParams[1].get<double>();
      const auto pLambda = gateParams[2].get<double>();
      cache.cMat = c_matrix_t({{std::cos(pTheta), 0.0},
                               {-std::cos(pLambda) * std::sin(pTheta),
                                -std::sin(pLambda) * std::sin(pTheta)},
                               {std::cos(pPhi) * std::sin(pTheta),
                                std::sin(pPhi) * std::sin(pTheta)},
                               {std::cos(pPhi + pLambda) * std::cos(pTheta),
                                std::sin(pPhi + pLambda) * std::cos(pTheta)}});
      cache.isConvertibleToCMat = Convertible;
      return;
    }
    cache.isConvertibleToCMat = UnConvertible;
    return;
  }
  case gCX: {
    cache.cMat = MatrixCX_c;
    cache.isConvertibleToCMat = Convertible;
    return;
  }
  case gCZ: {
    cache.cMat = MatrixCZ_c;
    cache.isConvertibleToCMat = Convertible;
    return;
  }
  case gCP: {
    if (gateParams[0].isNot<double>()) {
      cache.isConvertibleToCMat = UnConvertible;
      return;
    }
    const auto phase = gateParams[0].get<double>();
    setConvertible({
        {1.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {std::cos(phase), std::sin(phase)},
    });
    return;
  }

  default:
    cache.isConvertibleToCMat = UnConvertible;
    return;
  }
}

void GateMatrix::computeAndCachePMat() const {
  if (cache.isConvertibleToCMat == Convertible) {
    const auto edgeSize = cache.cMat.edgeSize();
    cache.pMat = p_matrix_t(edgeSize);
    for (size_t i = 0; i < edgeSize * edgeSize; ++i)
      cache.pMat[i] = Polynomial(cache.cMat[i]);
    cache.isConvertibleToPMat = Convertible;
    return;
  }
  cache.pMat = matCvt_gp_to_p(gateKind, gateParams);
  cache.isConvertibleToPMat = Convertible;
  return;
}

namespace {

inline void computeSigMatAfresh(const GateMatrix::c_matrix_t& cMat,
                                double zeroTol,
                                double oneTol,
                                GateMatrix::sig_matrix_t& sigMat) {
  assert(sigMat.edgeSize() == 0);
  const auto edgeSize = cMat.edgeSize();
  sigMat = GateMatrix::sig_matrix_t(edgeSize);

  for (size_t i = 0; i < edgeSize * edgeSize; ++i) {
    const auto& cplx = cMat[i];
    if (std::abs(cplx.real()) <= zeroTol)
      sigMat[i].real(cast::SK_Zero);
    else if (std::abs(cplx.real() - 1.0) <= oneTol)
      sigMat[i].real(cast::SK_One);
    else if (std::abs(cplx.real() + 1.0) <= oneTol)
      sigMat[i].real(cast::SK_MinusOne);

    if (std::abs(cplx.imag()) <= zeroTol)
      sigMat[i].imag(cast::SK_Zero);
    else if (std::abs(cplx.imag() - 1.0) <= oneTol)
      sigMat[i].imag(cast::SK_One);
    else if (std::abs(cplx.imag() + 1.0) <= oneTol)
      sigMat[i].imag(cast::SK_MinusOne);
  }
}

} // anonymous namespace

/// TODO: when there exists cached sigMat already, we can update sigMat more
/// efficiently
void GateMatrix::computeAndCacheSigMat(double zeroTol, double oneTol) const {
  const auto* cMat = getConstantMatrix();
  assert(cMat);
  assert(cMat->edgeSize() > 0);
  computeSigMatAfresh(*cMat, zeroTol, oneTol, cache.sigMat);
  cache.sigZeroTol = zeroTol;
  cache.sigOneTol = oneTol;
}

GateMatrix GateMatrix::lmatmul(const GateMatrix& other) const {
  auto* cMatA = getConstantMatrix();
  auto* cMatB = other.getConstantMatrix();
  assert(cMatA != nullptr && cMatB != nullptr);
  assert(cMatA->edgeSize() == cMatB->edgeSize());
  const auto edgeSize = cMatA->edgeSize();
  c_matrix_t cMat(edgeSize);

  // (BA)_ij = B_ik * A_kj
  for (size_t i = 0; i < edgeSize; ++i) {
    for (size_t j = 0; j < edgeSize; ++j) {
      cMat.rc(i, j) = {0.0, 0.0};
      for (size_t k = 0; k < edgeSize; ++k)
        cMat.rc(i, j) += cMatB->rc(i, k) * cMatA->rc(k, j);
    }
  }
  return GateMatrix(cMat);
}