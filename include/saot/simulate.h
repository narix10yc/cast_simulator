#ifndef SAOT_SIMULATE_H
#define SAOT_SIMULATE_H

#include "saot/CircuitGraph.h"
#include "saot/QuantumGate.h"
#include <iomanip>

namespace saot::simulate {

inline size_t insertZeroBit(size_t number, int index) {
  size_t left, right;
  left = (number >> index) << index;
  right = number - left;
  return (left << 1) ^ right;
}

template<typename real_t = double>
static void applyGeneral(std::complex<real_t>* sv, const saot::GateMatrix& gate,
                         const std::vector<int>& qubits, unsigned nqubits) {
  assert(gate.nqubits == qubits.size());
  assert(gate.isConstantMatrix());
  const auto& constMatrix = gate.matrix.constantMatrix.data;
  const auto& K = gate.N;

  // std::cerr << "applyGeneral (nqubits = " << nqubits << ") on qubits ";
  // for (const auto& q : qubits)
  //     std::cerr << q << " ";
  // std::cerr << "\nwith gate\n";
  // gate.printMatrix(std::cerr);

  std::vector<size_t> qubitsPower;
  for (const auto& q : qubits)
    qubitsPower.push_back(1 << q);

  auto qubitsSorted = qubits;
  std::sort(qubitsSorted.begin(), qubitsSorted.end());

  std::vector<size_t> idxVector(K);
  using complex_t = std::complex<real_t>;
  std::vector<complex_t> updatedAmp(K);

  for (size_t t = 0; t < (1 << (nqubits - gate.nqubits)); t++) {
    // extract indices
    for (size_t i = 0; i < K; i++) {
      size_t idx = t;
      for (const auto q : qubitsSorted)
        idx = insertZeroBit(idx, q);
      for (unsigned q = 0; q < gate.nqubits; q++) {
        if ((i & (1 << q)) > 0)
          idx |= (1 << qubits[q]);
      }
      idxVector[i] = idx;
    }

    // std::cerr << "t = " << t << ": [";
    // for (const auto& idx : idxVector)
    //     std::cerr << std::bitset<4>(idx) << ",";
    // std::cerr << "]\n";

    // multiply
    for (size_t i = 0; i < K; i++) {
      updatedAmp[i] = {0.0, 0.0};
      for (size_t ii = 0; ii < K; ii++)
        updatedAmp[i] +=
            sv[idxVector[ii]] * static_cast<complex_t>(constMatrix[i * K + ii]);
    }
    // store
    for (size_t i = 0; i < K; i++) {
      //     std::cerr << "idx " << i << " new amp = " << updatedAmp[i].real()
      //               << " + " << updatedAmp[i].imag() << "i"
      //               << " store back at position " << idxVector[i] << "\n";
      sv[idxVector[i]] = updatedAmp[i];
    }
  }
}

} // namespace saot::simulate

#endif // SAOT_SIMULATE_H