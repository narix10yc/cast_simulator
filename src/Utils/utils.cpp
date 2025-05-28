#include "utils/utils.h"
#include "utils/Formats.h"

bool utils::isPermutation(llvm::ArrayRef<int> arr) {
  std::vector<int> copy(arr.begin(), arr.end());
  std::ranges::sort(copy);
  for (unsigned i = 0, S = copy.size(); i < S; i++) {
    if (copy[i] != i)
      return false;
  }
  return true;
}

uint64_t utils::pdep64(uint64_t src, uint64_t mask, int nbits) {
  assert(0 <= nbits && nbits <= 64 && "nbits must be in [0, 64]");
  uint64_t dst = 0;
  unsigned k = 0;
  for (unsigned i = 0; i < nbits; i++) {
    if (mask & (1ULL << i)) {
      if (src & (1ULL << k))
        dst |= (1ULL << i);
      ++k;
    }
  }
  return dst;
}

uint32_t utils::pdep32(uint32_t src, uint32_t mask, int nbits) {
  assert(0 <= nbits && nbits <= 32 && "nbits must be in [0, 32]");
  uint32_t dst = 0;
  unsigned k = 0;
  for (unsigned i = 0; i < nbits; i++) {
    if (mask & (1U << i)) {
      if (src & (1U << k))
        dst |= (1U << i);
      ++k;
    }
  }
  return dst;
}

uint64_t utils::pext64(uint64_t src, uint64_t mask, int nbits) {
  assert(0 <= nbits && nbits <= 64 && "nbits must be in [0, 64]");
  uint64_t dst = 0;
  unsigned k = 0;
  for (unsigned i = 0; i < nbits; i++) {
    if (mask & (1ULL << i)) {
      if (src & (1ULL << i))
        dst |= (1 << k);
      ++k;
    }
  }

  return dst;
}

uint32_t utils::pext32(uint32_t src, uint32_t mask, int nbits) {
  assert(0 && "Not Implemented");
  return 0;
}

void utils::timedExecute(std::function<void()> f, const char* msg) {
  using clock = std::chrono::high_resolution_clock;
  const auto tic = clock::now();
  f();
  const auto tok = clock::now();
  const auto t_in_sec =
      std::chrono::duration_cast<std::chrono::microseconds>(tok - tic).count();
  std::cerr << "-- ("
            << fmt_time(static_cast<double>(t_in_sec) * 1e-6)
            << ") " << msg << "\n";
}

std::ostream& utils::print_complex(
    std::ostream& os, std::complex<double> c, int precision) {
  const double thres = 0.5 * std::pow(0.1, precision);
  if (c.real() >= -thres)
    os << " ";
  if (std::fabs(c.real()) < thres)
    os << "0." << std::string(precision, ' ');
  else
    os << std::fixed << std::setprecision(precision) << c.real();

  if (c.imag() >= -thres)
    os << "+";
  if (std::fabs(c.imag()) < thres)
    os << "0." << std::string(precision, ' ');
  else
    os << std::fixed << std::setprecision(precision) << c.imag();
  return os << "i";
}

std::complex<double> utils::inner_product(
    const std::complex<double>* aArr,
    const std::complex<double>* bArr,
    size_t length) {
  std::complex<double> s = 0.0;
  for (size_t i = 0; i < length; i++) {
    // (ar - i ai) * (br + i bi) = (ar*br + ai*bi) + i * (ar*bi - ai*br)
    s.real(
      s.real() + aArr[i].real() * bArr[i].real() +
      aArr[i].imag() * bArr[i].imag());
    s.imag(
      s.imag() + aArr[i].real() * bArr[i].imag() -
      aArr[i].imag() * bArr[i].real());
  }
  return s;
}

double utils::norm_squared(const std::complex<double>* arr, size_t len) {
  double s = 0.0;
  for (size_t i = 0; i < len; i++)
    s += arr[i].real() * arr[i].real() + arr[i].imag() * arr[i].imag();
  return s;
}