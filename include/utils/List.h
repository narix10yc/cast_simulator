#ifndef UTILS_LIST_H
#define UTILS_LIST_H

#include "utils/IteratorBase.h"
#include <cassert>
#include <memory>

namespace utils {

/// A double-linked list template
template <typename T> class List {
public:
  struct Node {
    T data;
    Node* prev;
    Node* next;
  }; // Node
private:
  template <typename _Iter, typename _Data> class IteratorBase {
    const _Iter* asIter() const { return static_cast<const _Iter*>(this); }
    _Iter* asIter() { return static_cast<_Iter*>(this); }

  public:
    _Data& operator*() const {
      assert(asIter()->current != nullptr);
      return asIter()->current->data;
    }

    _Data* operator->() const {
      assert(asIter()->current != nullptr);
      return &(asIter()->current->data);
    }

    bool operator==(std::nullptr_t) const {
      return asIter()->current == nullptr;
    }

    bool operator!=(std::nullptr_t) const {
      return asIter()->current != nullptr;
    }

    bool operator==(const _Iter& other) const {
      return asIter()->current == other.raw_ptr();
    }

    bool operator!=(const _Iter& other) const {
      return asIter()->current != other.raw_ptr();
    }

    _Iter& operator++() {
      assert(asIter()->current != nullptr && "Cannot (post-)increment");
      asIter()->current = asIter()->current->next;
      return *asIter();
    }

    _Iter operator++(int) {
      assert(asIter()->current != nullptr && "Cannot (pre-)increment");
      _Iter tmp = *asIter();
      asIter()->current = asIter()->current->next;
      return tmp;
    }

    _Iter next() const {
      assert(asIter()->current != nullptr && "Cannot increment");
      return _Iter(asIter()->current->next);
    }

    _Iter& operator--() {
      assert(asIter()->current != nullptr && "Cannot (post-)decrement");
      asIter()->current = asIter()->current->prev;
      return *asIter();
    }

    _Iter operator--(int) {
      assert(asIter()->current != nullptr && "Cannot (pre-)decrement");
      _Iter tmp = *asIter();
      asIter()->current = asIter()->current->prev;
      return tmp;
    }

    _Iter prev() const {
      assert(asIter()->current != nullptr && "Cannot decrement");
      return _Iter(asIter()->current->prev);
    }
  }; // IteratorBase

  class Iterator : public IteratorBase<Iterator, T> {
    Node* current;
    friend class IteratorBase<Iterator, T>;

  public:
    Iterator() : current(nullptr) {}
    explicit Iterator(Node* node) : current(node) {}

    /// Get the raw pointer of the \c Node* object so to access methods such as
    /// \c prev() and \c next() .
    Node* raw_ptr() const { return current; }
  }; // Iterator

  class ConstIterator : public IteratorBase<ConstIterator, const T> {
    Node* current;
    friend class IteratorBase<ConstIterator, const T>;

  public:
    explicit ConstIterator(Node* node) : current(node) {}
    ConstIterator(const Iterator& iter) : current(iter.raw_ptr()) {}

    /// Get the raw pointer of the \c Node* object so to access methods such as
    /// \c prev() and \c next() .
    Node* raw_ptr() const { return current; }
  }; // ConstIterator

  Node* _head;
  Node* _tail;
  size_t _size;

public:
  using iterator = Iterator;
  using const_iterator = ConstIterator;

  List() : _head(nullptr), _tail(nullptr), _size(0) {}

  ~List() { clear(); }

  List(const List& other) {
    for (const auto& data : other)
      push_back(data);
  }

  List(List&& other) noexcept
      : _head(other._head), _tail(other._tail), _size(other._size) {
    other._head = nullptr;
    other._tail = nullptr;
  }

  List& operator=(const List& other) {
    if (this == &other)
      return *this;
    this->~List();
    new (this) List(other);
    return *this;
  }

  List& operator=(List&& other) noexcept {
    if (this == &other)
      return *this;
    this->~List();
    new (this) List(std::move(other));
    return *this;
  }

  Node* head() { return _head; }
  const Node* head() const { return _head; }

  iterator head_iter() { return iterator(_head); }
  const_iterator head_iter() const { return const_iterator(_head); }
  const_iterator head_citer() const { return const_iterator(_head); }

  Node* tail() { return _tail; }
  const Node* tail() const { return _tail; }

  iterator tail_iter() { return iterator(_tail); }
  const_iterator tail_iter() const { return const_iterator(_tail); }
  const_iterator tail_citer() const { return const_iterator(_tail); }

  bool empty() const { return _size == 0; }

  size_t size() const { return _size; }

  void push_front(const T& value) { insert(_head, value); }

  void push_back(const T& value) { insert(nullptr, value); }

  void clear() {
    while (_head != nullptr)
      pop_front();
  }

  /// construct and insert an element to the front of the list
  template <typename... Args> void emplace_front(Args&&... args) {
    ++_size;
    Node* node = static_cast<Node*>(::operator new(sizeof(Node)));
    new (&node->data) T(std::forward<Args>(args)...); // placement new
    node->prev = nullptr;
    node->next = _head;
    if (_tail == nullptr)
      _tail = node;
    else
      _head->prev = node;
    _head = node;
  }

  /// construct and insert an element to the back of the list
  template <typename... Args> void emplace_back(Args&&... args) {
    ++_size;
    Node* node = static_cast<Node*>(::operator new(sizeof(Node)));
    new (&node->data) T(std::forward<Args>(args)...); // placement new
    node->prev = _tail;
    node->next = nullptr;
    if (_head == nullptr) {
      assert(_tail == nullptr);
      assert(_size == 1);
      _head = node;
    } else
      _tail->next = node;
    _tail = node;
  }

  /// destruct the first element and pop it from the list
  void pop_front() {
    assert(!empty());
    if (empty()) // safeguard
      return;
    --_size;
    auto* tmp = _head;
    _head = tmp->next;
    tmp->data.~T();
    ::operator delete(tmp);
    if (_head == nullptr) {
      assert(empty());
      _tail = nullptr;
    } else
      _head->prev = nullptr;
  }

  /// destruct the last element and pop it from the list
  void pop_back() {
    assert(!empty());
    if (empty()) // safeguard
      return;
    --_size;
    auto* tmp = _tail;
    _tail = tmp->prev;
    tmp->data.~T();
    ::operator delete(tmp);
    if (_tail == nullptr) {
      assert(empty());
      _head = nullptr;
    } else
      _tail->next = nullptr;
  }

  iterator begin() { return iterator(_head); }
  iterator end() { return iterator(nullptr); }

  const_iterator begin() const { return const_iterator(_head); }
  constexpr const_iterator end() const { return const_iterator(nullptr); }

  const_iterator cbegin() const { return const_iterator(_head); }
  constexpr const_iterator cend() const { return const_iterator(nullptr); }

  /// Before:  At - At.next
  /// After:   At - Node - At.next
  Node* insert(Node* at, const T& value) {
    assert(at != nullptr && "Cannot insert at NULL");
    ++_size;
    Node* node = static_cast<Node*>(::operator new(sizeof(Node)));
    new (&node->data) T(value); // copy constructor
    node->prev = at;
    node->next = at->next;
    if (at->next != nullptr)
      at->next->prev = node;
    at->next = node;
    return node;
  }

  iterator insert(const_iterator it, const T& value) {
    return iterator(insert(it.raw_ptr(), value));
  }

  /// Before:  At - At.next
  /// After:   At - Node - At.next
  template <typename... Args> Node* emplace_insert(Node* at, Args&&... args) {
    assert(at != nullptr && "Cannot insert at NULL");
    ++_size;
    Node* node = static_cast<Node*>(::operator new(sizeof(Node)));
    new (&node->data) T(std::forward<Args>(args)...); // placement new
    node->prev = at;
    node->next = at->next;
    if (at->next != nullptr)
      at->next->prev = node;
    at->next = node;
    return node;
  }

  template <typename... Args>
  iterator emplace_insert(const_iterator it, Args&&... args) {
    return iterator(emplace_insert(it.raw_ptr(), std::forward<Args>(args)...));
  }

  /// Before: At.prev - At - At.next
  /// After:  At.prev - At.next
  Node* erase(Node* at) {
    assert(at != nullptr && "Cannot erase at NULL");
    --_size;
    if (at->prev != nullptr)
      at->prev->next = at->next;
    if (at->next != nullptr)
      at->next->prev = at->prev;
    if (at == _head)
      _head = at->next;
    if (at == _tail)
      _tail = at->prev;
    auto* tmp = at->next;
    at->data.~T();
    ::operator delete(at);
    return tmp;
  }

  iterator erase(const_iterator it) { return iterator(erase(it.raw_ptr())); }
};

} // namespace utils

#endif // UTILS_LIST_H
