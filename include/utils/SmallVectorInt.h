#ifndef CAST_ADT_SMALL_VECTOR_INT
#define CAST_ADT_SMALL_VECTOR_INT

template <std::size_t NumInline> class SmallVectorInt {
  static_assert(sizeof(int) * NumInline >= 2 * sizeof(std::size_t));
  alignas(std::max_align_t) std::byte _data[sizeof(int) * NumInline];
  std::size_t _size;

  int* inlineData() { return reinterpret_cast<int*>(_data); }

  int* heapData() { return *reinterpret_cast<int**>(_data); }

  bool isInline() const { return _size <= NumInline; }

public:
  SmallVectorInt() : _size(0) {}

  ~SmallVectorInt() {
    if (!isInline())
      std::free(heapData());
  }

  int& operator[](std::size_t index) {
    assert(index < _size && "Index out of bounds");
    return (isInline() ? inlineData() : heapData())[index];
  }

  void push_back(int v) {
    if (_size < NumInline) {
      inlineData()[_size++] = v;
      return;
    }

    // dynamically allocated memory
    int** dataPtr = reinterpret_cast<int**>(_data);
    std::size_t curCapacity = capacity();
    if (_size == curCapacity) {
      // double the capacity
      int* newPtr =
          static_cast<int*>(std::malloc(2 * curCapacity * sizeof(int)));
      std::memcpy(newPtr, _data, curCapacity * sizeof(int));
      if (_size > NumInline)
        std::free(*dataPtr);
      *dataPtr = newPtr;
      std::size_t* capacityPtr =
          reinterpret_cast<std::size_t*>(_data + sizeof(std::size_t));
      *capacityPtr = 2 * curCapacity;
    }
    (*dataPtr)[_size++] = v;
  }

  std::size_t size() const { return _size; }

  std::size_t capacity() const {
    return isInline() ? NumInline
                      : *reinterpret_cast<const std::size_t*>(
                            _data + sizeof(std::size_t));
  }
};

#endif // CAST_ADT_SMALL_VECTOR_INT