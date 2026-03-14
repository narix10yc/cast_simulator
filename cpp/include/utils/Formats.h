#ifndef UTILS_FORMATS_H
#define UTILS_FORMATS_H

#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <type_traits>

namespace utils {

/// Writes precisely N digits of an unsigned integer into the provided buffer,
/// zero-padded. Intentionally trauncate if the value has more than N digits.
/// For example, write_uint<5>(buf, 123) will write "00123" into buf, while
/// write_uint<3>(buf, 123456) will write "456".
template <typename UInt, size_t N> void write_uint(char* out, UInt value) {
  static_assert(std::is_unsigned<UInt>::value);

  // compilers should be smart enough
  for (size_t i = 0; i < N; ++i) {
    char r = value % 10;
    value = value / 10;
    out[N - 1 - i] = static_cast<char>('0' + r);
  }
}

/// Writes a *non-negative* fixed-point number into the provided buffer. This
/// function writes (I + D + 1) characters, where I is the number of integer
/// digits, D is the number of decimal digits, and 1 is for the decimal point.
/// The value is rounded to D decimal places. The integer part is zero-padded
/// and truncated (internally uses write_uint). For example,
/// `write_fp<5,2>(buf, 123.456)` will write "00123.46" while
/// `write_fp<3,2>(buf, 12345.678)` will write "345.68". The decimal part is
/// rounded.
/// Notice that we should avoid truncating the integer part due to limited
/// precision of FP numbers.
template <typename FP, size_t I, size_t D> void write_fp(char* out, FP value) {
  static_assert(std::is_floating_point_v<FP>);
  assert(value >= 0.0);

  using UInt = uint64_t;

  UInt scaler = 1;
  for (size_t i = 0; i < D; ++i) {
    scaler *= 10;
  }

  // Round to nearest integer (value is always posible)
  UInt ivalue = static_cast<UInt>(value * scaler + 0.5);

  UInt ipart = ivalue / scaler;
  UInt dpart = ivalue % scaler;

  write_uint<UInt, I>(out, ipart);
  out[I] = '.';
  write_uint<UInt, D>(out + I + 1, dpart);
}

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

/// Fixed-width formatter base (CRTP).
///
/// Derived classes should implement:
///   void fmt(char*) const;
/// which fills the provided buffer (of size N) with the formatted string.
template <class Derived, size_t N> class fmt_fixed_width {
public:
  using buf_t = std::array<char, N>;

  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  friend std::ostream& operator<<(std::ostream& os, const fmt_fixed_width& b)
    requires requires(const Derived& d, char* c) {
      { d.fmt(c) } -> std::same_as<void>;
    }
  {
    char buf[N];
    b.derived().fmt(buf);
    os.write(buf, static_cast<std::streamsize>(N));
    return os;
  }
}; // class fmt_fixed_width

/// Format a number in the range [1.0, 1000.0) to a string with specified
/// width. For example, fmt_1_to_1e3<5>(123.45678) will print "123.5",
template <size_t N = 5>
class fmt_1_to_1e3 : public fmt_fixed_width<fmt_1_to_1e3<N>, N> {
  using base_t = fmt_fixed_width<fmt_1_to_1e3<N>, N>;
  double n;

public:
  explicit fmt_1_to_1e3(double n) : n(n) {
    static_assert(N > 4);
    assert(n >= 1.0 && n < 1000.0);
  }

  void fmt(char* out) const {
    assert(n >= 1.0 && n < 1000.0);

    if (n < 10.0) {
      write_fp<double, 1, N - 2>(out, n);
      return;
    }
    if (n < 100.0) {
      write_fp<double, 2, N - 3>(out, n);
      return;
    }
    // n < 1000.0
    write_fp<double, 3, N - 4>(out, n);
    return;
  }
}; // class fmt_1_to_1e3

/* Format a memory quantity into exactly 9 characters.
 *
 * Examples:
 *       0   B
 *       1   B
 *      12   B
 *     123   B
 *    1234   B
 *   12.35 KiB
 *   123.4 KiB
 *   1.235 MiB
 *   12.35 MiB
 *   123.5 MiB
 *   1.235 GiB
 *   ...
 *   123.5 TiB
 *   > 1.0 PiB
 */
class fmt_mem : public fmt_fixed_width<fmt_mem, 9> {
  size_t bytes;

public:
  explicit fmt_mem(size_t bytes) : bytes(bytes) {}

  void fmt(char* out) const {
    // Special case
    if (bytes < 10000) {
      out[5] = ' ';
      out[6] = ' ';
      out[7] = ' ';
      out[8] = 'B';
      if (bytes < 10) {
        out[0] = ' ';
        out[1] = ' ';
        out[2] = ' ';
        out[3] = ' ';
        out[4] = static_cast<char>('0' + bytes);
        return;
      }
      if (bytes < 100) {
        out[0] = ' ';
        out[1] = ' ';
        out[2] = ' ';
        write_uint<size_t, 2>(out + 3, bytes);
        return;
      }
      if (bytes < 1000) {
        out[0] = ' ';
        out[1] = ' ';
        write_uint<size_t, 3>(out + 2, bytes);
        return;
      }
      // bytes < 10_000
      out[0] = ' ';
      write_uint<size_t, 4>(out + 1, bytes);
      return;
    }

    if (bytes >= (1ULL << 50) * 1024) {
      std::memcpy(out, "> 1.0 PiB", 9);
      return;
    }

    double value;
    out[5] = ' ';
    // out[6] will be written later
    out[7] = 'i';
    out[8] = 'B';

    if (bytes < (1ULL << 20)) {
      // KiB
      value = static_cast<double>(bytes) / 1024.0;
      out[6] = 'K';
    } else if (bytes < (1ULL << 30)) {
      // MiB
      value = static_cast<double>(bytes) / (1ULL << 20);
      out[6] = 'M';
    } else if (bytes < (1ULL << 40)) {
      // GiB
      value = static_cast<double>(bytes) / (1ULL << 30);
      out[6] = 'G';
    } else {
      // TiB
      assert(bytes < (1ULL << 50));
      value = static_cast<double>(bytes) / (1ULL << 40);
      out[6] = 'T';
    }

    if (value >= 1000.0) {
      // [1000.0, 1024.0)
      write_fp<double, 4, 0>(out, value);
    } else {
      // [1.0, 1000.0)
      assert(1.0 <= value && value < 1000.0);
      fmt_1_to_1e3<5>(value).fmt(out);
    }
  }
}; // class fmt_mem

/*
 * Format a time duration (in seconds) into a fixed-width, 8-character string.
 * Special cases:
 *   - Values < 1 ns are formatted as "< 1.0 ns"
 *   - Values >= 100000 s are formatted as ">99999 s"
 *
 * Examples:
 *   < 1.0 ns
 *   1.235 ns
 *   12.35 ns
 *   123.5 ns
 *   1.235 us
 *   12.35 us
 *   123.5 us
 *   1.235 ms
 *   12.35 ms
 *   123.5 ms
 *   1.235  s
 *   12.35  s
 *   123.5  s
 *    1234  s
 *   12345  s
 *   >99999 s
 */
class fmt_time : public fmt_fixed_width<fmt_time, 8> {
  using base_t = fmt_fixed_width<fmt_time, 8>;

  double t_in_sec;

public:
  explicit fmt_time(double t_in_sec) : t_in_sec(t_in_sec) {
    assert(t_in_sec >= 0.0);
  }

  void fmt(char* out) const {
    // Special cases
    if (t_in_sec < 1e-9) {
      std::memcpy(out, "< 1.0 ns", 8);
      return;
    }

    if (t_in_sec >= 1e5) {
      std::memcpy(out, ">99999 s", 8);
      return;
    }

    double value;

    out[5] = ' ';
    out[6] = ' ';
    out[7] = 's';
    if (t_in_sec >= 1.0) {
      // seconds. early return
      if (t_in_sec < 1000.0) {
        fmt_1_to_1e3<5>(t_in_sec).fmt(out);
        return;
      }
      if (t_in_sec < 10000.0) {
        out[0] = ' ';
        write_uint<uint64_t, 4>(out + 1, static_cast<uint64_t>(t_in_sec));
        return;
      }
      // t_in_sec < 100_000 (asserted)
      write_uint<uint64_t, 5>(out, static_cast<uint64_t>(t_in_sec));
      return;
    }

    if (t_in_sec < 1e-6) {
      // nanoseconds
      value = t_in_sec * 1e9;
      out[6] = 'n';
    } else if (t_in_sec < 1e-3) {
      // microseconds
      value = t_in_sec * 1e6;
      out[6] = 'u';
    } else {
      assert(t_in_sec < 1.0);
      // milliseconds
      value = t_in_sec * 1e3;
      out[6] = 'm';
    }

    assert(1.0 <= value && value < 1000.0);
    fmt_1_to_1e3<5>(value).fmt(out);
  }
}; // class fmt_time

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