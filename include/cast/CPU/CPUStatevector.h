#ifndef CAST_CPU_CPU_STATEVECTOR_H
#define CAST_CPU_CPU_STATEVECTOR_H

#include "cast/Legacy/QuantumGate.h"
#include "cast/Core/QuantumGate.h"
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
    real = (ScalarType*)aligned_alloc(64, N * sizeof(ScalarType));
    imag = (ScalarType*)aligned_alloc(64, N * sizeof(ScalarType));
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
  int simd_s;
  int _nQubits;
  ScalarType* _data;
public:

  CPUStatevector(int nQubits, int simd_s, bool init = false)
    : simd_s(simd_s)
    , _nQubits(nQubits)
    , _data(static_cast<ScalarType*>(
      ::operator new((2ULL << nQubits) * sizeof(ScalarType),
      static_cast<std::align_val_t>(64)))) {
    if (init)
      initialize();
  }

  CPUStatevector(const CPUStatevector&) = delete;

  CPUStatevector(CPUStatevector&&) = delete;

  ~CPUStatevector() { ::operator delete(_data); }

  CPUStatevector& operator=(const CPUStatevector& that) {
    if (this == &that)
      return *this;
    std::memcpy(_data, that._data, sizeInBytes());
    return *this;
  }

  CPUStatevector& operator=(CPUStatevector&&) = delete;

  ScalarType* data() { return _data; }
  const ScalarType* data() const { return _data; }

  int nQubits() const { return _nQubits; }

  size_t getN() const { return 1ULL << _nQubits; }

  size_t sizeInBytes() const { return (2ULL << _nQubits) * sizeof(ScalarType); }

  ScalarType normSquared() const {
    ScalarType s = 0;
    for (size_t i = 0; i < 2 * getN(); i++)
      s += _data[i] * _data[i];
    return s;
  }

  ScalarType norm() const { return std::sqrt(normSquared()); }

  /// @brief Initialize to the |00...00> state.
  void initialize(int nThreads = 1) {
    std::memset(_data, 0, sizeInBytes());
    _data[0] = 1.0;
  }

  void normalize(int nThreads = 1) {
    ScalarType n = norm();
    for (size_t i = 0; i < 2 * getN(); ++i)
      _data[i] /= n;
  }

  /// @brief Uniform randomize statevector (by the Haar-measure on sphere).
  void randomize(int nThreads = 1) {
    auto N = getN();
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<ScalarType> d(0.0, 1.0);

    if (nThreads == 1) {
      for (size_t i = 0; i < 2 * N; ++i)
        _data[i] = d(gen);
      normalize();
      return;
    }
    // multi-thread
    std::vector<std::thread> threads;
    threads.reserve(nThreads);
    size_t nTasksPerThread = 2ULL * N / nThreads;
    for (int t = 0; t < nThreads; ++t) {
      size_t t0 = nTasksPerThread * t;
      size_t t1 = (t == nThreads - 1) ? 2ULL * N : nTasksPerThread * (t + 1);
      threads.emplace_back([&, t0=t0, t1=t1]() {
        for (size_t i = t0; i < t0; ++i)
          _data[i] = d(gen);
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
  const ScalarType& real(size_t idx) const {
    return _data[utils::insertZeroToBit(idx, simd_s)];
  }
  const ScalarType& imag(size_t idx) const {
    return _data[utils::insertOneToBit(idx, simd_s)];
  }

  std::complex<ScalarType> amp(size_t idx) const {
    size_t tmp = utils::insertZeroToBit(idx, simd_s);
    return { _data[tmp], _data[tmp | (1 << simd_s)] };
  }

  std::ostream& print(std::ostream& os = std::cerr) const {
    auto N = getN();
    if (N > 32) {
      os << BOLDCYAN("Warning: ")
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
  ScalarType prob(int q) const {
    ScalarType p = 0.0;
    for (size_t i = 0; i < (getN() >> 1); i++) {
      size_t idx = utils::insertZeroToBit(i, q);
      const auto& re = real(idx);
      const auto& im = imag(idx);
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

}; // class StatevectorAlt

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

// template<typename ScalarType>
// static ScalarType fidelity(const StatevectorSep<ScalarType>& sep, const
// StatevectorAlt<ScalarType>& alt) {
//     assert(sep.nQubits == alt.nQubits);

//     ScalarType re = 0.0, im = 0.0;
//     for (size_t i = 0; i < sep.N; i++) {
//         re += ( sep.real[i] * alt.data[2*i] + sep.imag[i] * alt.data[2*i+1]);
//         im += (-sep.real[i] * alt.data[2*i+1] + sep.imag[i] * alt.data[2*i]);
//     }
//     return re * re + im * im;
// }

// template<typename ScalarType>
// ScalarType fidelity(const StatevectorAlt<ScalarType>& alt, const
// StatevectorSep<ScalarType>& sep) {
//     return fidelity(sep, alt);
// }

// template<typename ScalarType>
// void StatevectorSep<ScalarType>::copyValueFrom(const StatevectorAlt<ScalarType>& alt)
// {
//     assert(nQubits == alt.nQubits);

//     for (size_t i = 0; i < N; i++) {
//         real[i] = alt.data[2*i];
//         imag[i] = alt.data[2*i+1];
//     }
// }

// template<typename ScalarType>
// void StatevectorAlt<ScalarType>::copyValueFrom(const StatevectorSep<ScalarType>& sep)
// {
//     assert(nQubits == sep.nQubits);

//     for (size_t i = 0; i < N; i++) {
//         data[2*i] = sep.real[i];
//         data[2*i+1] = sep.imag[i];
//     }
// }

} // namespace utils

#endif // CAST_CPU_CPU_STATEVECTOR_H