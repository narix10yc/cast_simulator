#include "cast/CPU/CPUStatevector.h"
#include "utils/Formats.h"
#include <bit>
#include <random>
#include <thread>
#include "utils/iocolor.h"

using namespace cast;

template <typename ScalarType>
CPUStatevector<ScalarType>::CPUStatevector(int nQubits,
                                           CPUSimdWidth simdWidth) {
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

// Copy constructor
template <typename ScalarType>
CPUStatevector<ScalarType>::CPUStatevector(const CPUStatevector& other) {
  nQubits_ = other.nQubits_;
  simd_s = other.simd_s;
  data_ = allocate();
  std::memcpy(data_, other.data_, sizeInBytes());
}

// Move constructor
template <typename ScalarType>
CPUStatevector<ScalarType>::CPUStatevector(CPUStatevector&& other) noexcept {
  if (this == &other)
    return;
  nQubits_ = other.nQubits_;
  simd_s = other.simd_s;
  data_ = other.data_;
  other.data_ = nullptr; // Prevent double deletion
}

// Copy assignment operator
template <typename ScalarType>
CPUStatevector<ScalarType>&
CPUStatevector<ScalarType>::operator=(const CPUStatevector& that) {
  if (this == &that)
    return *this;
  std::memcpy(data_, that.data_, sizeInBytes());
  return *this;
}

// Move assignment operator
template <typename ScalarType>
CPUStatevector<ScalarType>&
CPUStatevector<ScalarType>::operator=(CPUStatevector&& other) noexcept {
  if (this == &other)
    return *this;
  ::operator delete(data_);
  nQubits_ = other.nQubits_;
  simd_s = other.simd_s;
  data_ = other.data_;
  other.data_ = nullptr; // Prevent double deletion
  return *this;
}

template <typename ScalarType>
double CPUStatevector<ScalarType>::normSquared(int nThreads) const {
  if (nThreads <= 0)
    nThreads = cast::get_cpu_num_threads();

  double sum = 0.0;
  std::vector<std::thread> threads;
  threads.reserve(nThreads);
  std::vector<double> partialSums(nThreads, 0.0);
  auto N = getN();
  size_t nTasksPerThread = 2ULL * N / nThreads;
  for (int t = 0; t < nThreads; ++t) {
    size_t t0 = nTasksPerThread * t;
    size_t t1 = (t == nThreads - 1) ? 2ULL * N : nTasksPerThread * (t + 1);
    threads.emplace_back([this, t0, t1, p = partialSums.data() + t]() {
      double localSum = 0.0;
      for (size_t i = t0; i < t1; ++i)
        localSum += data_[i] * data_[i];
      *p = localSum;
    });
  }
  for (auto& t : threads) {
    if (t.joinable())
      t.join();
  }
  for (const auto& s : partialSums)
    sum += s;
  return sum;
}

template <typename ScalarType>
void CPUStatevector<ScalarType>::initialize(int nThreads) {
  if (nThreads <= 0)
    nThreads = cast::get_cpu_num_threads();

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

template <typename ScalarType>
void CPUStatevector<ScalarType>::normalize(int nThreads) {
  if (nThreads <= 0)
    nThreads = cast::get_cpu_num_threads();

  auto factor = 1.0 / norm(nThreads);
  std::vector<std::thread> threads;
  threads.reserve(nThreads);
  auto N = getN();
  size_t nTasksPerThread = 2ULL * N / nThreads;
  for (int t = 0; t < nThreads; ++t) {
    size_t t0 = nTasksPerThread * t;
    size_t t1 = (t == nThreads - 1) ? 2ULL * N : nTasksPerThread * (t + 1);
    threads.emplace_back([this, t0, t1, factor]() {
      for (size_t i = t0; i < t1; ++i)
        data_[i] *= factor;
    });
  }
  for (auto& t : threads) {
    if (t.joinable())
      t.join();
  }
}

template <typename ScalarType>
void CPUStatevector<ScalarType>::randomize(int nThreads) {
  if (nThreads <= 0)
    nThreads = cast::get_cpu_num_threads();

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

template <typename ScalarType>
std::ostream& CPUStatevector<ScalarType>::display(std::ostream& os) const {
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

template <typename ScalarType>
CPUStatevector<ScalarType>& CPUStatevector<ScalarType>::applyGate(
    const cast::StandardQuantumGate& stdQuGate) {
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
        ampUpdated[r] += static_cast<std::complex<ScalarType>>(mat.rc(r, c)) *
                         this->amp(ampIndices[c]);
      }
    }
    for (unsigned r = 0; r < K; r++) {
      this->real(ampIndices[r]) = ampUpdated[r].real();
      this->imag(ampIndices[r]) = ampUpdated[r].imag();
    }
  }
  return *this;
}

template <typename ScalarType>
std::vector<uint64_t> CPUStatevector<ScalarType>::sample(unsigned nShots,
                                                         uint64_t flag) const {

  std::vector<uint64_t> results;
  results.reserve(nShots);

  // zero out irrelevant bits
  flag = flag & ((1ULL << nQubits_) - 1);
  auto k = std::popcount(flag);

  std::vector<double> cdf(size_t(1) << k, 0.0);

  // compute CDF
  // TODO: use PDEP/PEXT
  for (size_t i = 0; i < getN(); i++) {
    size_t idx = 0;
    size_t bitPos = 0;

    for (size_t q = 0; q < nQubits_; q++) {
      if (flag & (1ULL << q)) {
        if (i & (1ULL << q)) {
          idx |= (1ULL << bitPos);
        }
        bitPos++;
      }
    }
    const double re = real(i);
    const double im = imag(i);
    cdf[idx] += (re * re + im * im);
  }

  // accumulate
  for (size_t i = 1, K = cdf.size(); i < K; ++i) {
    cdf[i] += cdf[i-1];
  }

  // Sample nShots times
  std::random_device rd;
  std::mt19937 gen{rd()};
  std::uniform_real_distribution<double> dis(0.0, 1.0);   

  return results;
}

namespace cast {
template class CPUStatevector<float>;
template class CPUStatevector<double>;
} // namespace cast