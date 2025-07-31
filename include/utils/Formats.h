#ifndef UTILS_FORMATS_H
#define UTILS_FORMATS_H

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>

namespace utils {

// fmt_0b: helper class to print binary of an integer
// uses: 'std::cerr << fmt_0b(123, 12)' to print 12 LSB of integer 123.
class fmt_0b {
  uint64_t v;
  int nbits;

public:
  fmt_0b(uint64_t v, int nbits) : v(v), nbits(nbits) {
    assert(nbits >= 0 && nbits <= 64);
  }

  friend std::ostream& operator<<(std::ostream& os, const fmt_0b& n) {
    for (int i = n.nbits - 1; i >= 0; --i)
      os.put((n.v & (1 << i)) ? '1' : '0');
    return os;
  }
}; // class fmt_0b

class fmt_complex {
  double re, im;
  int precision;

public:
  fmt_complex(float re, float im, int precision = 3)
      : re(static_cast<double>(re)), im(static_cast<double>(im)),
        precision(precision) {
    assert(precision >= 0 && precision <= 15);
  }

  fmt_complex(double re, double im, int precision = 3)
      : re(re), im(im), precision(precision) {
    assert(precision >= 0 && precision <= 15);
  }

  friend std::ostream& operator<<(std::ostream& os, const fmt_complex& c) {
    const double thres = 0.5 * std::pow(0.1, c.precision);
    if (c.re >= -thres)
      os << " ";
    if (std::fabs(c.re) < thres)
      os << "0." << std::string(c.precision, ' ');
    else
      os << std::fixed << std::setprecision(c.precision) << c.re;

    if (c.im >= -thres)
      os << "+";
    if (std::fabs(c.im) < thres)
      os << "0." << std::string(c.precision, ' ');
    else
      os << std::fixed << std::setprecision(c.precision) << c.im;
    return os << "i";
  }
}; // class fmt_complex

// Format time with 4 significant digits. For example, fmt_time(0.001234)
// will print "1.234 ms".
class fmt_time {
  double t_in_sec;

public:
  explicit fmt_time(double t_in_sec) : t_in_sec(t_in_sec) {
    assert(t_in_sec >= 0.0);
  }

  friend std::ostream& operator<<(std::ostream& os, const fmt_time& fmt) {
    // seconds
    if (fmt.t_in_sec >= 1e3)
      return os << static_cast<unsigned>(fmt.t_in_sec) << " s";
    if (fmt.t_in_sec >= 1e2)
      return os << std::fixed << std::setprecision(1) << fmt.t_in_sec << " s";
    if (fmt.t_in_sec >= 1e1)
      return os << std::fixed << std::setprecision(2) << fmt.t_in_sec << " s";
    if (fmt.t_in_sec >= 1.0)
      return os << std::fixed << std::setprecision(3) << fmt.t_in_sec << " s";
    // milliseconds
    if (fmt.t_in_sec >= 1e-1)
      return os << std::fixed << std::setprecision(1) << 1e3 * fmt.t_in_sec
                << " ms";
    if (fmt.t_in_sec >= 1e-2)
      return os << std::fixed << std::setprecision(2) << 1e3 * fmt.t_in_sec
                << " ms";
    if (fmt.t_in_sec >= 1e-3)
      return os << std::fixed << std::setprecision(3) << 1e3 * fmt.t_in_sec
                << " ms";
    // microseconds
    if (fmt.t_in_sec >= 1e-4)
      return os << std::fixed << std::setprecision(1) << 1e6 * fmt.t_in_sec
                << " us";
    if (fmt.t_in_sec >= 1e-5)
      return os << std::fixed << std::setprecision(2) << 1e6 * fmt.t_in_sec
                << " us";
    if (fmt.t_in_sec >= 1e-6)
      return os << std::fixed << std::setprecision(3) << 1e6 * fmt.t_in_sec
                << " us";
    // nanoseconds
    if (fmt.t_in_sec >= 1e-7)
      return os << std::fixed << std::setprecision(1) << 1e9 * fmt.t_in_sec
                << " ns";
    if (fmt.t_in_sec >= 1e-8)
      return os << std::fixed << std::setprecision(2) << 1e9 * fmt.t_in_sec
                << " ns";
    if (fmt.t_in_sec >= 1e-9)
      return os << std::fixed << std::setprecision(3) << 1e9 * fmt.t_in_sec
                << " ns";
    return os << "< 1.0 ns";
  }
}; // class fmt_time

/// Format a number in the range [1.0, 1000.0) to a string with specified width.
/// For example, fmt_1_to_1e3(123.45678, 5) will print "123.5",
class fmt_1_to_1e3 {
  double number;
  int width;

public:
  explicit fmt_1_to_1e3(double n, int width = 5) : number(n), width(width) {
    assert(n >= 0.0 &&
           "fmt_1_to_1e3: Currently only supporting positive numbers");
    assert(width >= 4);
    // assert(n >= 1.0 && n <= 1e3);
  }

  friend std::ostream& operator<<(std::ostream& os, const fmt_1_to_1e3& fmt) {
    if (fmt.number >= 100.0)
      return os << std::fixed << std::setprecision(fmt.width - 4) << fmt.number;
    if (fmt.number >= 10.0)
      return os << std::fixed << std::setprecision(fmt.width - 3) << fmt.number;
    return os << std::fixed << std::setprecision(fmt.width - 2) << fmt.number;
  }
};

template <typename T> class fmt_span {
  std::span<T> s;
  char separator;

public:
  fmt_span(std::span<T> s, char separator = ',') : s(s), separator(separator) {}

  friend std::ostream& operator<<(std::ostream& os, const fmt_span& fmt) {
    auto size = fmt.s.size();
    if (size == 0)
      return os << "[]";
    if (size == 1)
      return os << "[" << fmt.s[0] << "]";
    os << "[" << fmt.s[0];
    for (size_t i = 1; i < size; ++i) {
      os.put(fmt.separator);
      os << fmt.s[i];
    }
    return os << "]";
  }
};

} // namespace utils

#endif // UTILS_FORMATS_H