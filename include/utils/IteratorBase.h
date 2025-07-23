#ifndef UTILS_ITERATORBASE_H
#define UTILS_ITERATORBASE_H

namespace utils::impl {

/// CRTP class for iterators.
/// Derived may implement three methods:
/// _Iter& increment(), for operator++() and operator++(int);
/// _Iter& decrement(), for operator--() and operator--(int);
/// bool equals(const Derived&) const, for operator== and operator!=
template <typename _Iter, typename T> class IteratorBase {
public:
  // pre-increment
  _Iter& operator++() {
    static_cast<_Iter*>(this)->increment();
    return *static_cast<_Iter*>(this);
  }

  // post-increment
  _Iter operator++(int) {
    _Iter tmp = *static_cast<_Iter*>(this);
    static_cast<_Iter*>(this)->increment();
    return tmp;
  }

  // pre-decrement
  _Iter& operator--() {
    static_cast<_Iter*>(this)->decrement();
    return *static_cast<_Iter*>(this);
  }

  // post-decrement
  _Iter operator--(int) {
    _Iter tmp = *static_cast<_Iter*>(this);
    static_cast<_Iter*>(this)->decrement();
    return tmp;
  }

  // bool operator==(const _Iter& other) const {
  // return static_cast<const _Iter*>(this)->equals(other);
  // }

  // bool operator!=(const _Iter &other) const { return !(*this == other); }
  //
  // explicit operator T*() const {
  //   return static_cast<const _Iter*>(this)->ptr();
  // }
  //
  // explicit operator const T*() const {
  //   return static_cast<const _Iter*>(this)->const_ptr();
  // }
  //
  // T* operator->() const {
  //   return static_cast<const _Iter*>(this)->ptr();
  // }
  //
  // const T* operator->() const {
  //   return static_cast<const _Iter*>(this)->const_ptr();
  // }

}; // IteratorBase

} // namespace utils::impl

#endif // UTILS_ITERATORBASE_H
