#ifndef CAST_ADT_SMALL_VECTOR_INT
#define CAST_ADT_SMALL_VECTOR_INT

template <size_t NumInline> class SmallVectorInt {
  static_assert(sizeof(int) * NumInline >= 2 * sizeof(size_t));
  alignas(std::max_align_t) std::byte _data[sizeof(int) * NumInline];
  size_t _size;

  int* inlineData() { return reinterpret_cast<int*>(_data); }

  int* heapData() { return *reinterpret_cast<int**>(_data); }

  bool isInline() const { return _size <= NumInline; }

public:
  SmallVectorInt() : _size(0) {}

  ~SmallVectorInt() {
    if (!isInline())
      std::free(heapData());
  }

  int& operator[](size_t index) {
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
    size_t curCapacity = capacity();
    if (_size == curCapacity) {
      // double the capacity
      int* newPtr =
          static_cast<int*>(std::malloc(2 * curCapacity * sizeof(int)));
      std::memcpy(newPtr, _data, curCapacity * sizeof(int));
      if (_size > NumInline)
        std::free(*dataPtr);
      *dataPtr = newPtr;
      size_t* capacityPtr = reinterpret_cast<size_t*>(_data + sizeof(size_t));
      *capacityPtr = 2 * curCapacity;
    }
    (*dataPtr)[_size++] = v;
  }

  size_t size() const { return _size; }

  size_t capacity() const {
    return isInline()
               ? NumInline
               : *reinterpret_cast<const size_t*>(_data + sizeof(size_t));
  }
};

#endif // CAST_ADT_SMALL_VECTOR_INT