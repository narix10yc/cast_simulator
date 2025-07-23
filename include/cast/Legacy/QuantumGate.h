#ifndef CAST_LEGACY_QUANTUM_GATE_H
#define CAST_LEGACY_QUANTUM_GATE_H

#include "cast/Legacy/GateMatrix.h"
#include <array>

namespace cast::legacy {

/// A gate accepts up to 3 parameters that is either
/// - int: represents variable in parametrized gates
/// - double: parameter value
class QuantumGate {
private:
  mutable double opCountCache = -1.0;

public:
  /// The canonical form of qubits is in ascending order
  llvm::SmallVector<int> qubits;
  GateMatrix gateMatrix;

  QuantumGate() : qubits(), gateMatrix() {}

  QuantumGate(const GateMatrix& gateMatrix, int q)
      : qubits({q}), gateMatrix(gateMatrix) {
    assert(gateMatrix.nQubits() == 1);
  }

  QuantumGate(GateMatrix&& gateMatrix, int q)
      : qubits({q}), gateMatrix(gateMatrix) {
    assert(gateMatrix.nQubits() == 1);
  }

  QuantumGate(const GateMatrix& gateMatrix, std::initializer_list<int> qubits)
      : qubits(qubits), gateMatrix(gateMatrix) {
    assert(gateMatrix.nQubits() == qubits.size());
    sortQubits();
  }

  QuantumGate(GateMatrix&& gateMatrix, std::initializer_list<int> qubits)
      : qubits(qubits), gateMatrix(gateMatrix) {
    assert(gateMatrix.nQubits() == qubits.size());
    sortQubits();
  }

  QuantumGate(const GateMatrix& gateMatrix,
              const llvm::SmallVector<int>& qubits)
      : qubits(qubits), gateMatrix(gateMatrix) {
    assert(gateMatrix.nQubits() == qubits.size());
    sortQubits();
  }

  QuantumGate(GateMatrix&& gateMatrix, const llvm::SmallVector<int>& qubits)
      : qubits(qubits), gateMatrix(gateMatrix) {
    assert(gateMatrix.nQubits() == qubits.size());
    sortQubits();
  }

  int nQubits() const { return qubits.size(); }

  std::string getName() const { return GateKind2String(gateMatrix.gateKind); }

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
  QuantumGate lmatmul(const QuantumGate& other) const;

  double opCount(double zeroTol) const;

  bool isConvertibleToUnitaryPermGate(double tolerance) const {
    return gateMatrix.isConvertibleToUnitaryPermMatrix(tolerance);
  }

  bool isConvertibleToConstantGate() const {
    return gateMatrix.isConvertibleToConstantMatrix();
  }

  static QuantumGate I1(int q) {
    return QuantumGate(GateMatrix(GateMatrix::MatrixI1_c), q);
  }

  static QuantumGate I2(int q0, int q1) {
    return QuantumGate(GateMatrix(GateMatrix::MatrixI2_c), {q0, q1});
  }

  static QuantumGate H(int q) {
    return QuantumGate(GateMatrix(GateMatrix::MatrixH_c), q);
  }

  template <typename... Ints> static QuantumGate RandomUnitary(Ints... qubits) {
    static_assert((std::is_integral_v<Ints> && ...));
    constexpr auto nQubits = sizeof...(Ints);
    std::array<int, nQubits> qubitsCopy{qubits...};
    std::ranges::sort(qubitsCopy);
    return QuantumGate(
        GateMatrix(utils::randomUnitaryMatrix(1U << nQubits)),
        llvm::SmallVector<int>(qubitsCopy.begin(), qubitsCopy.end()));
  }

  template <std::size_t NQubits>
  static QuantumGate RandomUnitary(std::array<int, NQubits> qubits) {
    std::ranges::sort(qubits);
    return QuantumGate(GateMatrix(utils::randomUnitaryMatrix(1U << NQubits)),
                       llvm::SmallVector<int>(qubits.begin(), qubits.end()));
  }

  static QuantumGate RandomUnitary(const std::vector<int>& qubits) {
    llvm::SmallVector<int> sortedQubits(qubits.begin(), qubits.end());
    std::ranges::sort(sortedQubits);
    return QuantumGate(
        GateMatrix(utils::randomUnitaryMatrix(1U << qubits.size())),
        sortedQubits);
  }
};

} // namespace cast::legacy

#endif // CAST_LEGACY_QUANTUM_GATE_H