#ifndef CAST_CPU_CPU_STATEVECTOR_H
#define CAST_CPU_CPU_STATEVECTOR_H

#include "cast/CPU/Config.h"
#include "cast/Core/QuantumGate.h"
#include "utils/Formats.h"
#include "utils/iocolor.h"
#include "utils/utils.h"

#include <iostream>
#include <random>
#include <thread>
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

  CPUStatevector(int nQubits, CPUSimdWidth simdWidth) {
    assert(nQubits > 0);
    nQubits_ = nQubits;

    // initialize _data
    data_ = allocate();

    // initialize simd_s
    if constexpr (std::is_same_v<ScalarType, float>) {
      switch (simdWidth) {
      case CPUSimdWidth::W0:
        simd_s = 0;
        break; // 1 element
      case CPUSimdWidth::W128:
        simd_s = 2;
        break; // 4 elements
      case CPUSimdWidth::W256:
        simd_s = 3;
        break; // 8 elements
      case CPUSimdWidth::W512:
        simd_s = 4;
        break; // 16 elements
      default:
        assert(false && "Unsupported SIMD Width for float");
      }
    } else if constexpr (std::is_same_v<ScalarType, double>) {
      switch (simdWidth) {
      case CPUSimdWidth::W0:
        simd_s = 0;
        break; // 1 element
      case CPUSimdWidth::W128:
        simd_s = 1;
        break; // 2 elements
      case CPUSimdWidth::W256:
        simd_s = 2;
        break; // 4 elements
      case CPUSimdWidth::W512:
        simd_s = 3;
        break; // 8 elements
      default:
        assert(false && "Unsupported SIMD Width for double");
      }
    } else {
      static_assert(std::is_same_v<ScalarType, float> ||
                        std::is_same_v<ScalarType, double>,
                    "Unsupported ScalarType for CPUStatevector");
    }
  }

  CPUStatevector(const CPUStatevector& other) {
    nQubits_ = other.nQubits_;
    simd_s = other.simd_s;
    data_ = allocate();
    std::memcpy(data_, other.data_, sizeInBytes());
  }

  CPUStatevector(CPUStatevector&& other) noexcept {
    if (this == &other)
      return;
    nQubits_ = other.nQubits_;
    simd_s = other.simd_s;
    data_ = other.data_;
    other.data_ = nullptr; // Prevent double deletion
  }

  ~CPUStatevector() { ::operator delete(data_); }

  CPUStatevector& operator=(const CPUStatevector& that) {
    if (this == &that)
      return *this;
    std::memcpy(data_, that.data_, sizeInBytes());
    return *this;
  }

  CPUStatevector& operator=(CPUStatevector&& other) noexcept {
    if (this == &other)
      return *this;
    ::operator delete(data_);
    nQubits_ = other.nQubits_;
    simd_s = other.simd_s;
    data_ = other.data_;
    other.data_ = nullptr; // Prevent double deletion
    return *this;
  }

  ScalarType* data() { return data_; }
  const ScalarType* data() const { return data_; }

  int nQubits() const { return nQubits_; }

  size_t getN() const { return 1ULL << nQubits_; }

  size_t size() const { return 2ULL << nQubits_; }

  size_t sizeInBytes() const { return (2ULL << nQubits_) * sizeof(ScalarType); }

  double normSquared() const {
    double s = 0.0;
    for (size_t i = 0; i < 2 * getN(); i++) {
      double a = static_cast<double>(data_[i]);
      s += a * a;
    }
    return s;
  }

  double norm() const { return std::sqrt(normSquared()); }

  /// @brief Initialize to the |00...00> state.
  /// Notice: even though we provide nThreads parameter, this function
  /// uses a single-thread std::fill_n to initialize the statevector.
  void initialize(int nThreads = 1) {
    std::vector<std::thread> threads;
    threads.reserve(nThreads);
    auto N = getN();
    size_t nTasksPerThread = 2ULL * N / nThreads;
    for (int t = 0; t < nThreads; ++t) {
      size_t t0 = nTasksPerThread * t;
      size_t t1 = (t == nThreads - 1) ? 2ULL * N : nTasksPerThread * (t + 1);
      threads.emplace_back(
          [this, t0, t1]() { std::fill(data_ + t0, data_ + t1, 0.0); });
    }
    for (auto& t : threads) {
      if (t.joinable())
        t.join();
    }
    data_[0] = 1.0;
  }

  /// Notice: nThreads parameter is ignored in this function.
  void normalize(int nThreads = 1) {
    auto factor = 1.0 / norm();
    for (size_t i = 0; i < 2 * getN(); ++i)
      data_[i] *= factor;
  }

  /// @brief Uniform randomize statevector (by the Haar-measure on sphere).
  /// nThreads parameter does work here.
  void randomize(int nThreads = 1) {
    std::vector<std::thread> threads;
    threads.reserve(nThreads);
    auto N = getN();
    size_t nTasksPerThread = 2ULL * N / nThreads;
    for (int t = 0; t < nThreads; ++t) {
      size_t t0 = nTasksPerThread * t;
      size_t t1 = (t == nThreads - 1) ? 2ULL * N : nTasksPerThread * (t + 1);
      threads.emplace_back([this, t0, t1]() {
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::normal_distribution<ScalarType> d(0.0, 1.0);
        for (size_t i = t0; i < t1; ++i)
          this->data_[i] = d(gen);
      });
    }
    for (auto& t : threads) {
      if (t.joinable())
        t.join();
    }

    normalize(nThreads);
  }

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

  std::ostream& display(std::ostream& os = std::cerr) const {
    auto N = getN();
    if (N > 32) {
      os << BOLDCYAN("Info: ")
         << "statevector has more than 5 qubits, "
            "only the first 32 entries are shown.\n";
    }
    for (size_t i = 0; i < std::min<size_t>(32, N); i++) {
      os << utils::fmt_0b(i, 5) << " : " << utils::fmt_complex(real(i), imag(i))
         << "\n";
    }
    return os;
  }

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

  CPUStatevector& applyGate(const cast::StandardQuantumGate& stdQuGate) {
    const auto scalarGM = stdQuGate.getScalarGM();
    assert(scalarGM && "Can only apply constant gateMatrix");
    const auto& mat = scalarGM->matrix();

    const unsigned k = stdQuGate.nQubits();
    const unsigned K = 1 << k;
    assert(mat.edgeSize() == K);
    std::vector<size_t> ampIndices(K);
    std::vector<std::complex<ScalarType>> ampUpdated(K);

    size_t pdepMaskTask = ~static_cast<size_t>(0);
    size_t pdepMaskAmp = 0;
    for (const auto q : stdQuGate.qubits()) {
      pdepMaskTask ^= (1ULL << q);
      pdepMaskAmp |= (1ULL << q);
    }

    for (size_t taskId = 0; taskId < (getN() >> k); taskId++) {
      auto pdepTaskId = utils::pdep64(taskId, pdepMaskTask);
      for (size_t ampId = 0; ampId < K; ampId++) {
        ampIndices[ampId] = pdepTaskId + utils::pdep64(ampId, pdepMaskAmp);
      }

      // std::cerr << "taskId = " << taskId
      //           << " (" << utils::as0b(taskId, nQubits - k) << "):\n";
      // utils::printVectorWithPrinter(ampIndices,
      //   [&](size_t n, std::ostream& os) {
      //     os << n << " (" << utils::as0b(n, nQubits) << ")";
      //   }, std::cerr << " ampIndices: ") << "\n";

      for (unsigned r = 0; r < K; r++) {
        ampUpdated[r] = 0.0;
        for (unsigned c = 0; c < K; c++) {
          ampUpdated[r] += mat.rc(r, c) * this->amp(ampIndices[c]);
        }
      }
      for (unsigned r = 0; r < K; r++) {
        this->real(ampIndices[r]) = ampUpdated[r].real();
        this->imag(ampIndices[r]) = ampUpdated[r].imag();
      }
    }
    return *this;
  }

}; // class CPUStatevector

using CPUStatevectorFP32 = CPUStatevector<float>;
using CPUStatevectorFP64 = CPUStatevector<double>;

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

  void randomize(int nThreads = 1) {
    std::visit([nThreads](auto& s) { s.randomize(nThreads); }, sv);
  }

  void initialize(int nThreads = 1) {
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