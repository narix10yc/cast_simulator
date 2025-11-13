#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H

#include <cassert>
#include <complex>
#include <functional>
#include <iostream>
#include <span>
#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

namespace utils {

/// noexcept version of std::stoi for positive integers
/// Returns -1 for invalid input, -2 for overflow.
int parseNonNegativeInt(const char* str) noexcept;

/// @brief Sample K unique integers from [0, N-1] without replacement.
/// @param holder: will be cleared and filled with K unique integers.
void sampleNoReplacement(unsigned N, unsigned K, std::vector<int>& holder);

template <typename T>
bool isOrdered(const std::vector<T>& vec, bool ascending = true) {
  if (vec.empty())
    return true;

  if (ascending) {
    for (unsigned i = 0; i < vec.size() - 1; i++) {
      if (vec[i] > vec[i + 1])
        return false;
    }
    return true;
  } else {
    for (unsigned i = 0; i < vec.size() - 1; i++) {
      if (vec[i] < vec[i + 1])
        return false;
    }
    return true;
  }
}

std::ostream&
print_complex(std::ostream& os, std::complex<double> c, int precision = 3);

template <typename T>
void push_back_if_not_present(std::vector<T>& vec, const T& elem) {
  for (const auto& e : vec) {
    if (e == elem)
      return;
  }
  vec.push_back(elem);
}

template <typename T = uint64_t> T insertZeroToBit(T x, int bit) {
  T maskLo = (static_cast<T>(1) << bit) - 1;
  T maskHi = ~maskLo;
  return (x & maskLo) + ((x & maskHi) << 1);
}

template <typename T = uint64_t> T insertOneToBit(T x, int bit) {
  T maskLo = (static_cast<T>(1) << bit) - 1;
  T maskHi = ~maskLo;
  return (x & maskLo) | ((x & maskHi) << 1) | (1 << bit);
}

// parallel bit deposition
uint64_t pdep64(uint64_t src, uint64_t mask, unsigned nbits = 64);
uint32_t pdep32(uint32_t src, uint32_t mask, unsigned nbits = 32);

// parallel bit extraction
uint64_t pext64(uint64_t src, uint64_t mask, unsigned nbits = 64);
uint32_t pext32(uint32_t src, uint32_t mask, unsigned nbits = 32);

inline void displayProgressBar(float progress, int barWidth = 50) {
  // Clamp progress between 0 and 1
  assert(barWidth > 0);
  if (progress < 0.0f)
    progress = 0.0f;
  if (progress > 1.0f)
    progress = 1.0f;

  // Print the progress bar
  std::cerr.put('[');
  int i = 0;
  while (i < barWidth * progress) {
    std::cerr.put('=');
    ++i;
  }
  while (i < barWidth) {
    std::cerr.put(' ');
    ++i;
  }

  std::cerr << "] " << static_cast<int>(progress * 100.0f) << " %\r";
  std::cerr.flush();
}

inline void displayProgressBar(int nFinished, int nTotal, int barWidth = 50) {
  return displayProgressBar(static_cast<float>(nFinished) / nTotal, barWidth);
}

void timedExecute(std::function<void()> f, const char* msg);

/// @brief a dagger dotted with b
std::complex<double> inner_product(const std::complex<double>* aArrBegin,
                                   const std::complex<double>* bArrBegin,
                                   size_t length);

double norm_squared(const std::complex<double>* arrBegin, size_t length);

inline double norm(const std::complex<double>* arrBegin, size_t length) {
  return std::sqrt(norm_squared(arrBegin, length));
}

inline void normalize(std::complex<double>* arrBegin, size_t length) {
  double norm = utils::norm(arrBegin, length);
  for (size_t i = 0; i < length; i++)
    arrBegin[i] /= norm;
}

} // namespace utils

#endif // UTILS_UTILS_H