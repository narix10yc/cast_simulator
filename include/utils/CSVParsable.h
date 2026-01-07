#ifndef UTILS_CSVPARSABLE_H
#define UTILS_CSVPARSABLE_H

#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <concepts>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

namespace utils {

// CSVField: Every parsable type should provide a specialization of this struct
template <typename T> struct CSVField {
  static void parse(std::string_view token, T& field);
  static void write(std::ostream& os, const T& value);
};

template <typename T>
concept CSVFieldConcept =
    requires(std::string_view token, T& out, std::ostream& os, const T& in) {
      { CSVField<T>::parse(token, out) } -> std::same_as<void>;
      { CSVField<T>::write(os, in) } -> std::same_as<void>;
    };

// Specializations for common types
template <> struct CSVField<int> {
  static void parse(std::string_view token, int& field) {
    const char* first = token.data();
    const char* last = token.data() + token.size();
    auto [ptr, ec] = std::from_chars(first, last, field);
    (void)ptr;
    if (ec != std::errc{}) {
      // Keep behavior simple: fall back to 0 on parse failure.
      // (Callers typically validate input upstream.)
      field = 0;
    }
  }

  static void write(std::ostream& os, const int& value) { os << value; }
};

template <> struct CSVField<float> {
  static void parse(std::string_view token, float& field) {
    field = std::stof(std::string(token));
  }

  static void write(std::ostream& os, const float& value) { os << value; }
};

template <> struct CSVField<double> {
  static void parse(std::string_view token, double& field) {
    field = std::stod(std::string(token));
  }

  static void write(std::ostream& os, const double& value) { os << value; }
};

template <> struct CSVField<std::string> {
  static void parse(std::string_view token, std::string& field) {
    field = std::string(token);
  }

  static void write(std::ostream& os, const std::string& value) { os << value; }
};

template <> struct CSVField<std::string_view> {
  static void parse(std::string_view token, std::string_view& field) {
    field = token;
  }

  static void write(std::ostream& os, const std::string_view& value) {
    os << value;
  }
};

// -------------------- CSV Helper --------------------
inline std::vector<std::string_view> split_csv(std::string_view line) {
  std::vector<std::string_view> result;
  // Reserve: number of fields is commas + 1.
  result.reserve(
      1 + static_cast<size_t>(std::count(line.begin(), line.end(), ',')));
  size_t start = 0;
  size_t end = 0;

  while ((end = line.find(',', start)) != std::string_view::npos) {
    result.emplace_back(line.substr(start, end - start));
    start = end + 1;
  }
  result.emplace_back(line.substr(start)); // Add the last token
  return result;
}

// -------------------- Tuple Iteration --------------------
template <typename Tuple, typename Func, size_t... Is>
void for_each(Tuple&& t, Func&& f, std::index_sequence<Is...>) {
  (f(std::get<Is>(t), Is), ...); // Fold expression
}

template <typename Tuple, typename Func> void for_each(Tuple&& t, Func&& f) {
  constexpr auto size = std::tuple_size<std::decay_t<Tuple>>::value;
  for_each(std::forward<Tuple>(t),
           std::forward<Func>(f),
           std::make_index_sequence<size>{});
}

// -------------------- Base Class --------------------
template <typename Derived> struct CSVParsable {
  void parse(std::string_view line) {
    auto tokens = split_csv(line);
    assert(tokens.size() == Derived::num_fields);

    auto fields = static_cast<Derived*>(this)->tie_fields();
    for_each(fields, [&](auto& field, size_t i) {
      using FieldType = std::decay_t<decltype(field)>;
      static_assert(
          CSVFieldConcept<FieldType>,
          "CSVField<T> must provide: static void parse(std::string_view, T&) "
          "and static void write(std::ostream&, const T&)");
      CSVField<FieldType>::parse(tokens[i], field); // Use CSVField<T>::parse
    });
  }

  void write(std::ostream& os) const {
    auto fields = static_cast<const Derived*>(this)->tie_fields();
    bool first = true;
    for_each(fields, [&](const auto& field, size_t) {
      if (!first) {
        os << ",";
      }
      first = false;
      using FieldType = std::decay_t<decltype(field)>;
      static_assert(
          CSVFieldConcept<FieldType>,
          "CSVField<T> must provide: static void parse(std::string_view, T&) "
          "and static void write(std::ostream&, const T&)");
      CSVField<FieldType>::write(os, field); // Use CSVField<T>::write
    });
  }
};

// -------------------- Macro for Field Registration --------------------

// Helper for trim
consteval bool is_space(char c) { return c == ' ' || c == '\t'; }

// Trim the leading and trailing spaces and tabs from a string_view
consteval std::string_view trim(std::string_view str) {
  size_t start = 0;
  size_t end = str.size();

  while (start < end && is_space(str[start]))
    ++start;
  while (end > start && is_space(str[end - 1]))
    --end;

  return str.substr(start, end - start);
}

// Main consteval function to clean CSV title
template <size_t N>
consteval auto clean_csv_title(const char (&input)[N]) {
  std::array<char, N> out{};
  size_t out_i = 0;

  size_t token_start = 0;
  for (size_t i = 0; i < N; ++i) {
    if (input[i] == ',' || input[i] == '\0') {
      auto token = trim(std::string_view(&input[token_start], i - token_start));
      for (char c : token)
        out[out_i++] = c;

      // Add comma if not at the end
      if (input[i] == ',') {
        out[out_i++] = ',';
        token_start = i + 1;
      } else
        break;
    }
  }

  if (out_i < N) {
    out[out_i] = '\0';
  } else {
    out[N - 1] = '\0';
  }
  return out;
}

// Reflection is not available in C++20, so we use macros to register fields
// and generate the compile-time constant CSV_TITLE

#define CSV_STRINGIFY(...) #__VA_ARGS__

#define CSV_DATA_FIELD(...)                                                    \
  auto tie_fields() { return std::tie(__VA_ARGS__); }                          \
  auto tie_fields() const { return std::tie(__VA_ARGS__); }                    \
  static constexpr size_t num_fields =                                         \
      std::tuple_size<decltype(std::tie(__VA_ARGS__))>::value;                 \
  static constexpr auto CSV_TITLE_STORAGE =                                    \
      utils::clean_csv_title(CSV_STRINGIFY(__VA_ARGS__));                      \
  static constexpr std::string_view CSV_TITLE{CSV_TITLE_STORAGE.data()};

} // end namespace utils

#endif // UTILS_CSVPARSABLE_H