#ifndef CAST_UTILS_PRINT_SPAN_H
#define CAST_UTILS_PRINT_SPAN_H

#include <iostream>
#include <span>

namespace utils {

template <typename Printer_T, typename T>
concept Printer_C = requires(
    Printer_T printer, std::ostream& os, const T& value) {
  { printer(os, value) } -> std::same_as<void>;
};

template <typename T, typename Printer_T>
requires Printer_C<Printer_T, T>
std::ostream& printSpanWithPrinterNoBracket(
    std::ostream& os, std::span<T> v, Printer_T f) {
  auto it = v.begin();
  auto e = v.end();
  if (it == e)
    return os;
  std::invoke(f, os, *it);
  while (++it != e) {
    os << ",";
    std::invoke(f, os, *it);
  }
  return os;
}

template <typename T, typename Printer_T>
requires Printer_C<Printer_T, T>
std::ostream& printSpanWithPrinter(
    std::ostream& os, std::span<T> v, Printer_T f) {
  os.put('[');
  printSpanWithPrinterNoBracket(os, v, f);
  os.put(']');
  return os;
}

template<typename T>
std::ostream& printSpanNoBraket(std::ostream& os, std::span<T> v) {
  const auto printer = [](std::ostream& _os, const T& t) { _os << t; };
  return printSpanWithPrinterNoBracket<T, decltype(printer)>(os, v, printer);
}

template<typename T>
std::ostream& printSpan(std::ostream& os, std::span<T> v) {
  os.put('[');
  printSpanNoBraket(os, v);
  os.put(']');
  return os;
}

} // namespace utils

#endif // CAST_UTILS_PRINT_SPAN_H