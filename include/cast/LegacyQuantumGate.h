#ifndef CAST_QUANTUM_GATE_H
#define CAST_QUANTUM_GATE_H

#include "cast/LegacyGateMatrix.h"
#include <array>

namespace cast {

/// A gate accepts up to 3 parameters that is either
/// - int: represents variable in parametrized gates
/// - double: parameter value
class LegacyQuantumGate {
private:
  mutable double opCountCache = -1.0;

public:
  /// The canonical form of qubits is in ascending order
  llvm::SmallVector<int> qubits;
  LegacyGateMatrix gateMatrix;

  LegacyQuantumGate() : qubits(), gateMatrix() {}

  LegacyQuantumGate(const LegacyGateMatrix& gateMatrix, int q)
      : qubits({q}), gateMatrix(gateMatrix) {
    assert(gateMatrix.nQubits() == 1);
  }

  LegacyQuantumGate(LegacyGateMatrix&& gateMatrix, int q)
    : qubits({q}), gateMatrix(gateMatrix) {
    assert(gateMatrix.nQubits() == 1);
  }

  LegacyQuantumGate(const LegacyGateMatrix& gateMatrix, std::initializer_list<int> qubits)
      : qubits(qubits), gateMatrix(gateMatrix) {
    assert(gateMatrix.nQubits() == qubits.size());
    sortQubits();
  }

  LegacyQuantumGate(LegacyGateMatrix&& gateMatrix, std::initializer_list<int> qubits)
    : qubits(qubits), gateMatrix(gateMatrix) {
    assert(gateMatrix.nQubits() == qubits.size());
    sortQubits();
  }

  LegacyQuantumGate(const LegacyGateMatrix& gateMatrix, const llvm::SmallVector<int>& qubits)
    : qubits(qubits), gateMatrix(gateMatrix) {
    assert(gateMatrix.nQubits() == qubits.size());
    sortQubits();
  }

  LegacyQuantumGate(LegacyGateMatrix&& gateMatrix, const llvm::SmallVector<int>& qubits)
    : qubits(qubits), gateMatrix(gateMatrix) {
    assert(gateMatrix.nQubits() == qubits.size());
    sortQubits();
  }

  int nQubits() const { return qubits.size(); }

  std::string getName() const {
    return impl::GateKind2String(gateMatrix.gateKind);
  }

  bool isQubitsSorted() const {
    if (qubits.empty())
      return true;
    for (unsigned i = 0; i < qubits.size() - 1; i++) {
      if (qubits[i + 1] <= qubits[i])
        return false;
    }
    return true;
  }

  bool checkConsistency() const {
    return (gateMatrix.nQubits() == qubits.size());
  }

  std::ostream& displayInfo(std::ostream& os) const;

  int findQubit(int q) const {
    for (unsigned i = 0; i < qubits.size(); i++) {
      if (qubits[i] == q)
        return i;
    }
    return -1;
  }

  void sortQubits();

  /// @brief B.lmatmul(A) will return AB. That is, gate B will be applied first.
  LegacyQuantumGate lmatmul(const LegacyQuantumGate& other) const;

  double opCount(double zeroTol) const;

  bool isConvertibleToUnitaryPermGate(double tolerance) const {
    return gateMatrix.isConvertibleToUnitaryPermMatrix(tolerance);
  }

  bool isConvertibleToConstantGate() const {
    return gateMatrix.isConvertibleToConstantMatrix();
  }

  static LegacyQuantumGate I1(int q) {
    return LegacyQuantumGate(LegacyGateMatrix(LegacyGateMatrix::MatrixI1_c), q);
  }

  static LegacyQuantumGate I2(int q0, int q1) {
    return LegacyQuantumGate(LegacyGateMatrix(LegacyGateMatrix::MatrixI2_c), {q0, q1});
  }

  static LegacyQuantumGate H(int q) {
    return LegacyQuantumGate(LegacyGateMatrix(LegacyGateMatrix::MatrixH_c), q);
  }

  template<typename... Ints>
  static LegacyQuantumGate RandomUnitary(Ints... qubits) {
    static_assert((std::is_integral_v<Ints> && ...));
    constexpr auto nQubits = sizeof...(Ints);
    std::array<int, nQubits> qubitsCopy{qubits...};
    std::ranges::sort(qubitsCopy);
    return LegacyQuantumGate(
      LegacyGateMatrix(utils::randomUnitaryMatrix(1U << nQubits)),
      llvm::SmallVector<int>(qubitsCopy.begin(), qubitsCopy.end()));
  }

  template<std::size_t NQubits>
  static LegacyQuantumGate RandomUnitary(std::array<int, NQubits> qubits) {
    std::ranges::sort(qubits);
    return LegacyQuantumGate(
      LegacyGateMatrix(utils::randomUnitaryMatrix(1U << NQubits)),
      llvm::SmallVector<int>(qubits.begin(), qubits.end()));
  }

  static LegacyQuantumGate RandomUnitary(const std::vector<int>& qubits) {
    llvm::SmallVector<int> sortedQubits(qubits.begin(), qubits.end());
    std::ranges::sort(sortedQubits);
    return LegacyQuantumGate(
        LegacyGateMatrix(utils::randomUnitaryMatrix(1U << qubits.size())),
        sortedQubits);
  }

};

} // namespace cast

#endif // CAST_QUANTUM_GATE_H