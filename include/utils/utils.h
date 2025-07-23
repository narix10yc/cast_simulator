#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H

#include <algorithm>
#include <cassert>
#include <chrono>
#include <complex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace utils {

/// @brief Sample K unique integers from [0, N-1] without replacement.
/// @param holder: will be cleared and filled with K unique integers.
void sampleNoReplacement(unsigned N, unsigned K, std::vector<int>& holder);

bool isPermutation(llvm::ArrayRef<int> arr);

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
std::ostream& printArrayNoBrackets(std::ostream& os,
                                   llvm::ArrayRef<T> arr,
                                   const char sep = ',') {
  if (arr.empty())
    return os;
  auto it = arr.begin();
  os << *it;
  while (++it != arr.end()) {
    os.put(sep);
    os << *it;
  }
  return os;
}

template <typename T>
std::ostream&
printArray(std::ostream& os, llvm::ArrayRef<T> arr, const char sep = ',') {
  if (arr.empty())
    return os << "[]";
  auto it = arr.begin();
  os << "[" << *it;
  while (++it != arr.end()) {
    os.put(sep);
    os << *it;
  }
  return os << "]";
}

template <typename T>
std::ostream&
printArray(std::ostream& os, std::span<T> arr, const char sep = ',') {
  if (arr.empty())
    return os << "[]";
  auto it = arr.begin();
  os << "[" << *it;
  while (++it != arr.end()) {
    os.put(sep);
    os << *it;
  }
  return os << "]";
}

template <typename T, typename Printer>
std::ostream& printArray(std::ostream& os,
                         llvm::ArrayRef<T> arr,
                         Printer pFunc,
                         const char sep = ',') {
  if (arr.empty())
    return os << "[]";
  auto it = arr.begin();
  os << "[";
  pFunc(os, *it);
  while (++it != arr.end()) {
    os.put(sep);
    pFunc(os, *it);
  }
  return os << "]";
}

template <typename T, unsigned N>
std::ostream& printArray(std::ostream& os,
                         const llvm::SmallVector<T, N>& arr,
                         const char sep = ',') {
  return printArray(os, llvm::ArrayRef<T>(arr), sep);
}

// @param f: The printer is expected to take inputs (const T&, std::ostream&)
template <typename T, typename Printer>
std::ostream& printVectorWithPrinter(const std::vector<T>& v,
                                     Printer f,
                                     std::ostream& os = std::cerr) {
  if (v.empty())
    return os << "[]";
  auto it = v.cbegin();
  f(*it, os << "[");
  while (++it != v.cend())
    f(*it, os << ",");
  return os << "]";
}

template <typename T>
void push_back_if_not_present(std::vector<T>& vec, const T& elem) {
  for (const auto& e : vec) {
    if (e == elem)
      return;
  }
  vec.push_back(elem);
}

template <typename T, unsigned N>
void pushBackIfNotInVector(llvm::SmallVector<T, N>& vec, T elem) {
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
uint64_t pdep64(uint64_t src, uint64_t mask, int nbits = 64);
uint32_t pdep32(uint32_t src, uint32_t mask, int nbits = 32);

// parallel bit extraction
uint64_t pext64(uint64_t src, uint64_t mask, int nbits = 64);
uint32_t pext32(uint32_t src, uint32_t mask, int nbits = 32);

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