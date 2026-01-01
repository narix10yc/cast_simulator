#ifndef UTILS_FORMATS_H
#define UTILS_FORMATS_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Fixed-width formatter base (CRTP).
//
// Derived classes should implement:
//   buf_t fmt() const;
// where buf_t is std::array<char, N>.
template <class Derived, size_t N> class fmt_fixed_width {
public:
  using buf_t = std::array<char, N>;

  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  friend std::ostream& operator<<(std::ostream& os, const fmt_fixed_width& b)
    requires requires(const Derived& d) {
      { d.fmt() } -> std::same_as<std::array<char, N>>;
    }
  {
    const auto buf = b.derived().fmt();
    os.write(buf.data(), static_cast<std::streamsize>(N));
    return os;
  }
};

/// Format a memory quantity into exactly N characters.
/// Default N=9 yields strings like:
/// 512    -> "  512   B"
/// 1234   -> " 1234   B"
/// 12345  -> "12.35 MiB"
/// 123456 -> "123.5 MiB"
template <size_t N = 9> class fmt_mem : public fmt_fixed_width<fmt_mem<N>, N> {
  using base_t = fmt_fixed_width<fmt_mem<N>, N>;
  size_t bytes;

  static constexpr const char* unit_str(unsigned u) {
    // 3-char units to keep suffix width fixed.
    switch (u) {
    case 0:
      return "  B";
    case 1:
      return "KiB";
    case 2:
      return "MiB";
    case 3:
      return "GiB";
    default:
      return "TiB";
    }
  }

public:
  explicit fmt_mem(size_t bytes) : bytes(bytes) {}

  base_t::buf_t fmt() const {
    static_assert(N >= 8, "fmt_mem<N>: N must be >= 8");
    constexpr size_t k_suffix = 4; // " " + 3-char unit
    static_assert(N > k_suffix, "fmt_mem<N>: N too small");

    const int num_w = static_cast<int>(N - k_suffix);

    // Decimal scaling (1000-based) to match the examples.
    unsigned unit = 0;
    double value = static_cast<double>(bytes);
    while (unit < 4 && value >= 1000.0) {
      value /= 1000.0;
      ++unit;
    }

    // Choose decimals so the numeric field fits nicely into num_w.
    // We aim for ~4 significant digits, but never exceed the field.
    int decimals = 0;
    if (value >= 100.0)
      decimals = std::max(0, num_w - 4);
    else if (value >= 10.0)
      decimals = std::max(0, num_w - 3);
    else if (value >= 1.0)
      decimals = std::max(0, num_w - 2);
    else
      decimals = std::max(0, num_w - 1);

    // Format into a temporary buffer (N+1 for snprintf's terminator).
    char tmp[N + 1];
    std::memset(tmp, ' ', sizeof(tmp));
    tmp[N] = '\0';

    // Right-align numeric field into num_w, then " " + unit.
    const int written = std::snprintf(
        tmp, sizeof(tmp), "%*.*f %s", num_w, decimals, value, unit_str(unit));

    typename base_t::buf_t out{};
    if (written < 0) {
      out.fill('#');
      return out;
    }

    // Ensure exactly N characters (pad with spaces if snprintf produced fewer).
    for (size_t i = 0; i < N; ++i)
      out[i] = (tmp[i] == '\0') ? ' ' : tmp[i];

    return out;
  }
};

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

// Format a span of elements with a separator and square brakets. For example,
// std::cerr << fmt_span(std::span<int>{1, 2, 3}) will give "[1,2,3]"
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