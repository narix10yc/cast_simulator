/* Happy ChatGPT */

#ifndef CAST_ADT_SORTEDVECTORMAP_H
#define CAST_ADT_SORTEDVECTORMAP_H

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace cast {

template <typename Key,
          typename T,
          typename Compare = std::less<>, // transparent by default
          typename Alloc = std::allocator<std::pair<Key, T>>>
class SortedVectorMap {
public:
  using key_type = Key;
  using mapped_type = T;
  using value_type = std::pair<Key, T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using key_compare = Compare;
  using allocator_type = Alloc;

  using container_type = std::vector<value_type, Alloc>;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;

private:
  container_type data_;
  key_compare comp_{};

  template <class KLike>
  static bool eq_(const key_compare& comp, const Key& a, const KLike& b) {
    return !comp(a, b) && !comp(b, a);
  }

  template <class KLike> iterator lower_bound_impl(const KLike& k) {
    return std::lower_bound(data_.begin(),
                            data_.end(),
                            k,
                            [&](const value_type& kv, const KLike& key) {
                              return comp_(kv.first, key);
                            });
  }

  template <class KLike> const_iterator lower_bound_impl(const KLike& k) const {
    return std::lower_bound(data_.begin(),
                            data_.end(),
                            k,
                            [&](const value_type& kv, const KLike& key) {
                              return comp_(kv.first, key);
                            });
  }

public:
  // ——— ctors ———
  SortedVectorMap() = default;

  explicit SortedVectorMap(const key_compare& comp,
                           const allocator_type& alloc = allocator_type())
      : data_(alloc), comp_(comp) {}

  explicit SortedVectorMap(const allocator_type& alloc) : data_(alloc) {}

  SortedVectorMap(std::initializer_list<value_type> init,
                  const key_compare& comp = key_compare(),
                  const allocator_type& alloc = allocator_type())
      : data_(alloc), comp_(comp) {
    data_.assign(init.begin(), init.end());
    std::sort(data_.begin(),
              data_.end(),
              [&](const value_type& a, const value_type& b) {
                if (comp_(a.first, b.first))
                  return true;
                if (comp_(b.first, a.first))
                  return false;
                return false; // equal keys: keep first
              });
    // unique by key (last one wins, like map's insert-or-assign if duplicated)
    auto last = std::unique(data_.begin(),
                            data_.end(),
                            [&](const value_type& a, const value_type& b) {
                              return eq_(comp_, a.first, b.first);
                            });
    data_.erase(last, data_.end());
  }

  // ——— iterators ———
  iterator begin() noexcept { return data_.begin(); }
  const_iterator begin() const noexcept { return data_.begin(); }
  const_iterator cbegin() const noexcept { return data_.cbegin(); }
  iterator end() noexcept { return data_.end(); }
  const_iterator end() const noexcept { return data_.end(); }
  const_iterator cend() const noexcept { return data_.cend(); }

  // ——— capacity ———
  bool empty() const noexcept { return data_.empty(); }
  size_type size() const noexcept { return data_.size(); }
  size_type capacity() const noexcept { return data_.capacity(); }
  void reserve(size_type n) { data_.reserve(n); }
  void shrink_to_fit() { data_.shrink_to_fit(); }

  // ——— element access ———
  
  // operator[] inserts default-constructed value if missing
  T& operator[](const Key& k) {
    auto it = lower_bound_impl(k);
    if (it == end() || comp_(k, it->first)) {
      it = data_.insert(it, value_type(k, T{}));
    }
    return it->second;
  }
  T& operator[](Key&& k) {
    auto it = lower_bound_impl(k);
    if (it == end() || comp_(k, it->first)) {
      it = data_.insert(it, value_type(std::move(k), T{}));
    }
    return it->second;
  }

  // ——— lookup (heterogeneous if Compare is transparent) ———
  template <class KLike> iterator find(const KLike& k) {
    auto it = lower_bound_impl(k);
    if (it != end() && eq_(comp_, it->first, k))
      return it;
    return end();
  }
  template <class KLike> const_iterator find(const KLike& k) const {
    auto it = lower_bound_impl(k);
    if (it != end() && eq_(comp_, it->first, k))
      return it;
    return end();
  }

  template <class KLike> bool contains(const KLike& k) const {
    return find(k) != end();
  }

  template <class KLike> iterator lower_bound(const KLike& k) {
    return lower_bound_impl(k);
  }
  template <class KLike> const_iterator lower_bound(const KLike& k) const {
    return lower_bound_impl(k);
  }

  template <class KLike> iterator upper_bound(const KLike& k) {
    // first element with key > k
    return std::upper_bound(data_.begin(),
                            data_.end(),
                            k,
                            [&](const KLike& key, const value_type& kv) {
                              return comp_(key, kv.first);
                            });
  }
  template <class KLike> const_iterator upper_bound(const KLike& k) const {
    return std::upper_bound(data_.begin(),
                            data_.end(),
                            k,
                            [&](const KLike& key, const value_type& kv) {
                              return comp_(key, kv.first);
                            });
  }

  // ——— modifiers ———

  void clear() noexcept { data_.clear(); }

  // insert unique (fails if key exists). Returns {iterator, inserted}
  std::pair<iterator, bool> insert(const value_type& kv) {
    auto it = lower_bound_impl(kv.first);
    if (it != end() && eq_(comp_, it->first, kv.first)) {
      return {it, false};
    }
    return {data_.insert(it, kv), true};
  }

  std::pair<iterator, bool> insert(value_type&& kv) {
    auto it = lower_bound_impl(kv.first);
    if (it != end() && eq_(comp_, it->first, kv.first)) {
      return {it, false};
    }
    return {data_.insert(it, std::move(kv)), true};
  }

  template <class KArg, class... VArgs>
  std::pair<iterator, bool> try_emplace(KArg&& k, VArgs&&... vargs) {
    auto it = lower_bound_impl(k);
    if (it != end() && eq_(comp_, it->first, k)) {
      return {it, false};
    }
    return {data_.insert(it,
                         value_type(std::forward<KArg>(k),
                                    T(std::forward<VArgs>(vargs)...))),
            true};
  }

  template <class KArg, class VArg>
  std::pair<iterator, bool> insert_or_assign(KArg&& k, VArg&& v) {
    auto it = lower_bound_impl(k);
    if (it != end() && eq_(comp_, it->first, k)) {
      it->second = std::forward<VArg>(v);
      return {it, false};
    }
    return {data_.insert(
                it, value_type(std::forward<KArg>(k), std::forward<VArg>(v))),
            true};
  }

  iterator erase(const_iterator pos) { return data_.erase(pos); }

  template <class KLike> size_type erase(const KLike& k) {
    auto it = find(k);
    if (it == end())
      return 0;
    data_.erase(it);
    return 1;
  }

  void swap(SortedVectorMap& other) noexcept(
      std::allocator_traits<Alloc>::propagate_on_container_swap::value ||
      std::allocator_traits<Alloc>::is_always_equal::value) {
    data_.swap(other.data_);
    std::swap(comp_, other.comp_);
  }

  // ——— access to underlying storage ———
  const container_type& data() const noexcept { return data_; }
  container_type& data() noexcept { return data_; }

  // ——— comparators ———
  key_compare key_comp() const { return comp_; }

  // equality by content
  friend bool operator==(const SortedVectorMap& a, const SortedVectorMap& b) {
    return a.data_ == b.data_;
  }
  friend bool operator!=(const SortedVectorMap& a, const SortedVectorMap& b) {
    return !(a == b);
  }
};

} // namespace cast

#endif // CAST_ADT_SORTEDVECTORMAP_H