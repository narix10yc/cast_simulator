#ifndef CAST_CPU_CPU_STATEVECTOR_H
#define CAST_CPU_CPU_STATEVECTOR_H

#include "cast/Legacy/QuantumGate.h"
#include "cast/Core/QuantumGate.h"
#include "cast/CPU/Config.h"
#include "utils/iocolor.h"
#include "utils/utils.h"
#include "utils/TaskDispatcher.h"

#include <complex>
#include <iostream>
#include <random>
#include <thread>
#include <algorithm>
#include <cstdlib>

namespace cast {

template<typename ScalarType>
class StatevectorSep;

template<typename ScalarType>
class CPUStatevector;

template<typename ScalarType>
class StatevectorSep {
public:
  int nQubits;
  uint64_t N;
  ScalarType* real;
  ScalarType* imag;

  StatevectorSep(int nQubits, bool initialize = false)
      : nQubits(nQubits), N(1ULL << nQubits) {
    assert(nQubits > 0);
    real = (ScalarType*)std::aligned_alloc(64, N * sizeof(ScalarType));
    imag = (ScalarType*)std::aligned_alloc(64, N * sizeof(ScalarType));
    if (initialize) {
      for (size_t i = 0; i < (1 << nQubits); i++) {
        real[i] = 0;
        imag[i] = 0;
      }
      real[0] = 1.0;
    }
    // std::cerr << "StatevectorSep(int)\n";
  }

  StatevectorSep(const StatevectorSep& that)
      : nQubits(that.nQubits), N(that.N) {
    real = (ScalarType*)std::aligned_alloc(64, N * sizeof(ScalarType));
    imag = (ScalarType*)std::aligned_alloc(64, N * sizeof(ScalarType));
    for (size_t i = 0; i < that.N; i++) {
      real[i] = that.real[i];
      imag[i] = that.imag[i];
      // std::cerr << "StatevectorSep(const StatevectorSep&)\n";
    }
  }

  StatevectorSep(StatevectorSep&& that) noexcept
      : nQubits(that.nQubits), N(that.N), real(that.real), imag(that.imag) {
    that.real = nullptr;
    that.imag = nullptr;
    // std::cerr << "StatevectorSep(StatevectorSep&&)\n";
  }

  ~StatevectorSep() {
    std::free(real);
    std::free(imag);
    // std::cerr << "~StatevectorSep\n";
  }

  StatevectorSep& operator=(const StatevectorSep& that) {
    if (this != &that) {
      for (size_t i = 0; i < N; i++) {
        real[i] = that.real[i];
        imag[i] = that.imag[i];
      }
    }
    // std::cerr << "=(const StatevectorSep&)\n";
    return *this;
  }

  StatevectorSep& operator=(StatevectorSep&& that) noexcept {
    this->~StatevectorSep();
    real = that.real;
    imag = that.imag;
    nQubits = that.nQubits;
    N = that.N;

    that.real = nullptr;
    that.imag = nullptr;
    // std::cerr << "=(StatevectorSep&&)\n";
    return *this;
  }

  // void copyValueFrom(const StatevectorAlt<ScalarType>&);

  ScalarType normSquared(int nthreads = 1) const {
    const auto f = [&](uint64_t i0, uint64_t i1, ScalarType &rst) {
      ScalarType sum = 0.0;
      for (uint64_t i = i0; i < i1; i++) {
        sum += real[i] * real[i];
        sum += imag[i] * imag[i];
      }
      rst = sum;
    };

    if (nthreads == 1) {
      ScalarType s;
      f(0, N, s);
      return s;
    }

    std::vector<std::thread> threads(nthreads);
    std::vector<ScalarType> sums(nthreads);
    uint64_t blockSize = N / nthreads;
    for (uint64_t i = 0; i < nthreads; i++) {
      uint64_t i0 = i * blockSize;
      uint64_t i1 = (i == nthreads - 1) ? N : ((i + 1) * blockSize);
      threads[i] = std::thread(f, i0, i1, std::ref(sums[i]));
    }

    for (auto& thread : threads)
      thread.join();

    ScalarType sum = 0.0;
    for (const auto& s : sums)
      sum += s;
    return sum;
  }

  ScalarType norm(int nthreads = 1) const {
    return std::sqrt(normSquared(nthreads));
  }

  void normalize(int nthreads = 1) {
    ScalarType n = norm(nthreads);
    const auto f = [&](uint64_t i0, uint64_t i1) {
      for (uint64_t i = i0; i < i1; i++) {
        real[i] /= n;
        imag[i] /= n;
      }
    };

    if (nthreads == 1) {
      f(0, N);
      return;
    }
    std::vector<std::thread> threads(nthreads);
    uint64_t blockSize = N / nthreads;
    for (uint64_t i = 0; i < nthreads; i++) {
      uint64_t i0 = i * blockSize;
      uint64_t i1 = (i == nthreads - 1) ? N : ((i + 1) * blockSize);
      threads[i] = std::thread(f, i0, i1);
    }

    for (auto& thread : threads)
      thread.join();
  }

  void randomize(int nthreads = 1) {
    const auto f = [&](uint64_t i0, uint64_t i1) {
      std::random_device rd;
      std::mt19937 gen{rd()};
      std::normal_distribution<ScalarType> d{0, 1};
      for (uint64_t i = i0; i < i1; i++) {
        real[i] = d(gen);
        imag[i] = d(gen);
      }
    };

    if (nthreads == 1) {
      f(0, N);
      normalize(nthreads);
      return;
    }

    std::vector<std::thread> threads(nthreads);
    uint64_t blockSize = N / nthreads;
    for (uint64_t i = 0; i < nthreads; i++) {
      uint64_t i0 = i * blockSize;
      uint64_t i1 = (i == nthreads - 1) ? N : ((i + 1) * blockSize);
      threads[i] = std::thread(f, i0, i1);
    }

    for (auto& thread : threads)
      thread.join();
    normalize(nthreads);
  }

  std::ostream& print(std::ostream& os) const {
    if (N > 32) {
      os << IOColor::BOLD << IOColor::CYAN_FG << "Warning: " << IOColor::RESET
         << "statevector has more than 5 qubits, "
            "only the first 32 entries are shown.\n";
    }
    for (size_t i = 0; i < ((N > 32) ? 32 : N); i++) {
      os << i << ": ";
      utils::print_complex(os, {real[i], imag[i]});
      os << "\n";
    }
    return os;
  }
};


/// @brief CPUStatevector stores statevector in a single array with alternating
/// real and imaginary parts. The alternating pattern is controlled by
/// \p simd_s. More precisely, the memory is stored as an iteration of $2^s$
/// real parts followed by $2^s$ imaginary parts.
/// For example,
/// memory index: 000 001 010 011 100 101 110 111
/// amplitudes:   r00 r01 i00 i01 r10 r11 i10 i11
template<typename ScalarType>
class CPUStatevector {
private:
  ScalarType* _data;
  int _nQubits;
  int simd_s;

  // Make sure _nQubits is set.
  [[nodiscard]] inline ScalarType* allocate() {
    const size_t size = sizeInBytes();
    const auto align =
      static_cast<std::align_val_t>(std::min<size_t>(size, 64));
    return static_cast<ScalarType*>(::operator new(size, align));
  }
public:
  CPUStatevector(int nQubits, CPUSimdWidth simdWidth) {
    assert(nQubits > 0);
    _nQubits = nQubits;
    
    // initialize _data
    _data = allocate();

    // initialize simd_s
    if constexpr (std::is_same_v<ScalarType, float>) {
      switch (simdWidth) {
        case CPUSimdWidth::W128: simd_s = 2; break; // 4 elements
        case CPUSimdWidth::W256: simd_s = 3; break; // 8 elements
        case CPUSimdWidth::W512: simd_s = 4; break; // 16 elements
        default: assert(false && "Unsupported SIMD Width for float");
      }
    } else if constexpr (std::is_same_v<ScalarType, double>) {
      switch (simdWidth) {
        case CPUSimdWidth::W128: simd_s = 1; break; // 2 elements
        case CPUSimdWidth::W256: simd_s = 2; break; // 4 elements
        case CPUSimdWidth::W512: simd_s = 3; break; // 8 elements
        default: assert(false && "Unsupported SIMD Width for double");
      }
    } else {
      static_assert(std::is_same_v<ScalarType, float> ||
                    std::is_same_v<ScalarType, double>,
                    "Unsupported ScalarType for CPUStatevector");
    }
  }

  CPUStatevector(const CPUStatevector& other) {
    _nQubits = other._nQubits;
    simd_s = other.simd_s;
    _data = allocate();
    std::memcpy(_data, other._data, sizeInBytes());
  }

  CPUStatevector(CPUStatevector&& other) noexcept {
    if (this == &other)
      return;
    _nQubits = other._nQubits;
    simd_s = other.simd_s;
    _data = other._data;
    other._data = nullptr; // Prevent double deletion
  }

  ~CPUStatevector() { ::operator delete(_data); }

  CPUStatevector& operator=(const CPUStatevector& that) {
    if (this == &that)
      return *this;
    std::memcpy(_data, that._data, sizeInBytes());
    return *this;
  }

  CPUStatevector& operator=(CPUStatevector&& other) noexcept {
    if (this == &other)
      return *this;
    ::operator delete(_data);
    _nQubits = other._nQubits;
    simd_s = other.simd_s;
    _data = other._data;
    other._data = nullptr; // Prevent double deletion
    return *this;
  }

  ScalarType* data() { return _data; }
  const ScalarType* data() const { return _data; }

  int nQubits() const { return _nQubits; }

  size_t getN() const { return 1ULL << _nQubits; }

  size_t sizeInBytes() const { return (2ULL << _nQubits) * sizeof(ScalarType); }

  double normSquared() const {
    double s = 0.0;
    for (size_t i = 0; i < 2 * getN(); i++) {
      double a = static_cast<double>(_data[i]);
      s += a * a;
    }
    return s;
  }

  double norm() const { return std::sqrt<double>(normSquared()); }

  /// @brief Initialize to the |00...00> state.
  /// Notice: even though we provide nThreads parameter, this function
  /// uses a single-thread std::memset to initialize the statevector.
  void initialize(int nThreads = 1) {
    std::memset(_data, 0, sizeInBytes());
    _data[0] = 1.0;
  }

  /// Notice: nThreads parameter is ignored in this function.
  void normalize(int nThreads = 1) {
    ScalarType n = norm();
    for (size_t i = 0; i < 2 * getN(); ++i)
      _data[i] /= n;
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
          this->_data[i] = d(gen);
      });
    }
    for (auto& t : threads) {
      if (t.joinable())
        t.join();
    }

    normalize(nThreads);
  }

  ScalarType& real(size_t idx) {
    return _data[utils::insertZeroToBit(idx, simd_s)];
  }
  ScalarType& imag(size_t idx) {
    return _data[utils::insertOneToBit(idx, simd_s)];
  }
  ScalarType real(size_t idx) const {
    return _data[utils::insertZeroToBit(idx, simd_s)];
  }
  ScalarType imag(size_t idx) const {
    return _data[utils::insertOneToBit(idx, simd_s)];
  }

  std::complex<ScalarType> amp(size_t idx) const {
    size_t tmp = utils::insertZeroToBit(idx, simd_s);
    return { _data[tmp], _data[tmp | (1 << simd_s)] };
  }

  std::ostream& print(std::ostream& os = std::cerr) const {
    auto N = getN();
    if (N > 32) {
      os << BOLDCYAN("Info: ")
         << "statevector has more than 5 qubits, "
            "only the first 32 entries are shown.\n";
    }
    for (size_t i = 0; i < std::min<size_t>(32, N); i++) {
      os << i << ": ";
      utils::print_complex(os, {real(i), imag(i)}, 8);
      os << "\n";
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
    for (int q = 0; q < _nQubits; q++) {
      os << "qubit " << q << ": " << prob(q) << "\n";
    }
    return os;
  }

  CPUStatevector& applyGate(const cast::legacy::QuantumGate& gate) {
    const auto* cMat = gate.gateMatrix.getConstantMatrix();
    assert(cMat && "Can only apply constant gateMatrix");

    const unsigned k = gate.qubits.size();
    const unsigned K = 1 << k;
    assert(cMat->edgeSize() == K);
    std::vector<size_t> ampIndices(K);
    std::vector<std::complex<ScalarType>> ampUpdated(K);

    size_t pdepMaskTask = ~static_cast<size_t>(0);
    size_t pdepMaskAmp = 0;
    for (const auto q : gate.qubits) {
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
          ampUpdated[r] += cMat->rc(r, c) * this->amp(ampIndices[c]);
        }
      }
      for (unsigned r = 0; r < K; r++) {
        this->real(ampIndices[r]) = ampUpdated[r].real();
        this->imag(ampIndices[r]) = ampUpdated[r].imag();
      }
    }
    return *this;
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

using CPUStatevectorF32 = CPUStatevector<float>;
using CPUStatevectorF64 = CPUStatevector<double>;

// extern template class CPUStatevector<float>;
// extern template class CPUStatevector<double>;

// template<typename ScalarType>
// ScalarType fidelity(
//     const StatevectorSep<ScalarType>& sv1, const StatevectorSep<ScalarType>& sv2) {
//   assert(sv1.nQubits == sv2.nQubits);

//   ScalarType re = 0.0, im = 0.0;
//   for (size_t i = 0; i < sv1.N; i++) {
//     re += (sv1.real[i] * sv2.real[i] + sv1.imag[i] * sv2.imag[i]);
//     im += (-sv1.real[i] * sv2.imag[i] + sv1.imag[i] * sv2.real[i]);
//   }
//   return re * re + im * im;
// }

template<typename ScalarType>
ScalarType fidelity(
    const CPUStatevector<ScalarType>& sv0, const CPUStatevector<ScalarType>& sv1) {
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

using CPUStatevectorF32 = CPUStatevector<float>;
using CPUStatevectorF64 = CPUStatevector<double>;

} // end of namespace cast

#endif // CAST_CPU_CPU_STATEVECTOR_H