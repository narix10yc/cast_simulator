#ifndef UTILS_IS_POD_H
#define UTILS_IS_POD_H

#include <type_traits>

namespace utils {

template <typename T>
concept is_pod = std::is_trivially_copyable_v<T>;
// std::is_trivially_copyable_v<T> &&
// std::is_trivially_default_constructible_v<T> &&
// std::is_standard_layout_v<T>;
} // namespace utils

#endif // UTILS_IS_POD_H
