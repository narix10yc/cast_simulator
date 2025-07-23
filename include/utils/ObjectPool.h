#ifndef UTILS_OBJECTPOOL_H
#define UTILS_OBJECTPOOL_H

#include <cassert>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <unordered_set>

namespace utils {

/// Simple implementation of object pooling.
template <typename T, size_t BlockSize = 128> class ObjectPool {
  std::vector<T*> memoryBlocks;
  std::unordered_set<T*> acquired;
  std::deque<T*> availables;

  void extendPool() {
    T* ptr =
        static_cast<T*>(std::aligned_alloc(alignof(T), sizeof(T) * BlockSize));
    memoryBlocks.push_back(ptr);

    for (size_t i = 0; i < BlockSize; ++i)
      availables.push_back(ptr + i);
  }

public:
  ObjectPool() : memoryBlocks(), acquired(), availables() { extendPool(); }

  ~ObjectPool() {
    for (T* ptr : acquired)
      ptr->~T();
    for (auto* block : memoryBlocks)
      std::free(block);
    memoryBlocks.clear();
    acquired.clear();
    availables.clear();
  }

  ObjectPool(const ObjectPool&) = delete;
  ObjectPool& operator=(const ObjectPool&) = delete;
  ObjectPool(ObjectPool&&) = delete;
  ObjectPool& operator=(ObjectPool&&) = delete;

  /// acquire an instance from the pool and construct it in-place
  template <typename... Args> T* acquire(Args&&... args) {
    if (availables.empty())
      extendPool();

    assert(!availables.empty());
    T* ptr = availables.back();
    availables.pop_back();
    acquired.insert(ptr);
    ;

    // construct the object in-place
    new (ptr) T(std::forward<Args>(args)...);
    return ptr;
  }

  /// destroy an object and put back into the pool
  void release(T* ptr) {
    assert(acquired.find(ptr) != acquired.end());
    acquired.erase(ptr);
    ptr->~T();
    availables.push_back(ptr);
  }
};

} // namespace utils

#endif // UTILS_OBJECTPOOL_H
