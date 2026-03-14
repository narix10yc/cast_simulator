#ifndef CAST_CPU_CPU_STATEVECTOR_H
#define CAST_CPU_CPU_STATEVECTOR_H

#include "cast/CPU/Config.h"
#include "cast/Core/QuantumGate.h"
#include "utils/utils.h"

#include <iostream>
#include <variant>

namespace cast {

/// @brief CPUStatevector stores statevector in a single array with alternating
/// real and imaginary parts. The alternating pattern is controlled by
/// \p simd_s. More precisely, the memory is stored as an iteration of $2^s$
/// real parts followed by $2^s$ imaginary parts.
/// For example,
/// memory index: 000 001 010 011 100 101 110 111
/// amplitudes:   r00 r01 i00 i01 r10 r11 i10 i11
template <typename ScalarType> class CPUStatevector {
private:
  ScalarType* data_;
  int nQubits_;
  int simd_s;

  // Make sure _nQubits is set.
  [[nodiscard]] inline ScalarType* allocate() {
    const size_t size = sizeInBytes();
    const auto align =
        static_cast<std::align_val_t>(std::min<size_t>(size, 64));
    return static_cast<ScalarType*>(::operator new(size, align));
  }

public:
  CPUStatevector() : data_(nullptr), nQubits_(0), simd_s(0) {}

  CPUStatevector(int nQubits, CPUSimdWidth simdWidth);

  CPUStatevector(const CPUStatevector& other);
  CPUStatevector(CPUStatevector&& other) noexcept;
  CPUStatevector& operator=(const CPUStatevector& that);
  CPUStatevector& operator=(CPUStatevector&& other) noexcept;

  ~CPUStatevector() { ::operator delete(data_); }

  ScalarType* data() { return data_; }
  const ScalarType* data() const { return data_; }

  int nQubits() const { return nQubits_; }

  size_t getN() const { return 1ULL << nQubits_; }

  size_t size() const { return 2ULL << nQubits_; }

  size_t sizeInBytes() const { return (2ULL << nQubits_) * sizeof(ScalarType); }

  double normSquared(int nThreads = 0) const;

  double norm(int nThreads = 0) const {
    return std::sqrt(normSquared(nThreads));
  }

  /// @brief Initialize to the |00...00> state.
  /// Notice: even though we provide nThreads parameter, this function
  /// uses a single-thread std::fill_n to initialize the statevector.
  void initialize(int nThreads = 0);

  /// @brief Normalize the statevector.
  void normalize(int nThreads = 0);

  /// @brief Uniformly randomize statevector (by the Haar-measure on sphere).
  void randomize(int nThreads = 0);

  ScalarType& real(size_t idx) {
    return data_[utils::insertZeroToBit(idx, simd_s)];
  }
  ScalarType& imag(size_t idx) {
    return data_[utils::insertOneToBit(idx, simd_s)];
  }
  ScalarType real(size_t idx) const {
    return data_[utils::insertZeroToBit(idx, simd_s)];
  }
  ScalarType imag(size_t idx) const {
    return data_[utils::insertOneToBit(idx, simd_s)];
  }

  std::complex<ScalarType> amp(size_t idx) const {
    size_t tmp = utils::insertZeroToBit(idx, simd_s);
    return {data_[tmp], data_[tmp | (1 << simd_s)]};
  }

  std::ostream& display(std::ostream& os = std::cerr) const;

  /// Sample nShot measurements.
  /// flag indicates which qubits to measure (1 = measure, 0 = ignore).
  /// For example, for an 8-qubit statevector, flag = 0b00101010 means to
  /// measure qubits 1, 3, and 5. In this case, the returned vector contains
  /// nShots samples where in each sample, the lowest 3 bits correspond to
  /// qubits 1, 3, and 5.
  /// Sampling relies on building a partial CDF.
  std::vector<uint64_t> sample(unsigned nShots, uint64_t flag) const;

  /// @brief Compute the probability of measuring 1 on qubit q
  double prob(int q) const {
    double p = 0.0;
    for (size_t i = 0; i < (getN() >> 1); i++) {
      size_t idx = utils::insertZeroToBit(i, q);
      const double re = real(idx);
      const double im = imag(idx);
      p += (re * re + im * im);
    }
    return 1.0 - p;
  }

  std::ostream& printProbabilities(std::ostream& os = std::cerr) const {
    for (int q = 0; q < nQubits_; q++) {
      os << "qubit " << q << ": " << prob(q) << "\n";
    }
    return os;
  }

  CPUStatevector& applyGate(const cast::StandardQuantumGate& stdQuGate);
}; // class CPUStatevector

using CPUStatevectorFP32 = CPUStatevector<float>;
using CPUStatevectorFP64 = CPUStatevector<double>;

extern template class CPUStatevector<float>;
extern template class CPUStatevector<double>;

template <typename ScalarType>
ScalarType fidelity(const CPUStatevector<ScalarType>& sv0,
                    const CPUStatevector<ScalarType>& sv1) {
  assert(sv0.nQubits() == sv1.nQubits());
  ScalarType re = 0.0, im = 0.0;
  for (size_t i = 0; i < sv0.getN(); i++) {
    auto amp0 = sv0.amp(i);
    auto amp1 = sv1.amp(i);
    re += amp0.real() * amp1.real() + amp0.imag() * amp1.imag();
    im += amp0.real() * amp1.imag() - amp0.imag() * amp1.real();
  }
  return re * re + im * im;
}

// This is a wrapper around FP32 and FP64 CPU Statevectors. Internally just a
// std::variant.
class CPUStatevectorWrapper {
private:
  std::variant<CPUStatevectorFP32, CPUStatevectorFP64> sv;

public:
  CPUStatevectorWrapper(Precision precision,
                        int nQubits,
                        CPUSimdWidth simdWidth) {
    if (precision == Precision::FP32)
      sv = CPUStatevectorFP32(nQubits, simdWidth);
    else if (precision == Precision::FP64)
      sv = CPUStatevectorFP64(nQubits, simdWidth);
    else
      assert(false && "Unsupported precision for CPUStatevectorWrapper");
  }

  void* data() {
    return std::visit([](auto& s) { return static_cast<void*>(s.data()); }, sv);
  }

  void randomize(int nThreads = 0) {
    std::visit([nThreads](auto& s) { s.randomize(nThreads); }, sv);
  }

  void initialize(int nThreads = 0) {
    std::visit([nThreads](auto& s) { s.initialize(nThreads); }, sv);
  }

  size_t sizeInBytes() const {
    return std::visit([](const auto& s) { return s.sizeInBytes(); }, sv);
  }

  int nQubits() const {
    return std::visit([](const auto& s) { return s.nQubits(); }, sv);
  }

}; // class CPUStatevectorWrapper

} // end of namespace cast

#endif // CAST_CPU_CPU_STATEVECTOR_H