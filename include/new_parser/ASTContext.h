#ifndef CAST_NEW_PARSER_ASTCONTEXT_H
#define CAST_NEW_PARSER_ASTCONTEXT_H

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <new>
#include <vector>
#include <span>

#include "new_parser/AST.h"

namespace cast::draft {
namespace ast {

class ASTContext {
private:
  struct MemoryNode {
    static constexpr size_t Size = 4096; // 4KB
    std::byte* begin;
    std::byte* end;
    std::byte* current;
    MemoryNode* next;
  };

  class MemoryManager {
  private:
    MemoryNode* head;
    MemoryNode* tail;
    struct Overflow {
      std::byte* data;
      size_t size;
    };
    std::vector<Overflow> overflows;

    void allocateNewNode() {
      auto* newNode = new MemoryNode();
      newNode->begin = new std::byte[MemoryNode::Size];
      newNode->end = newNode->begin + MemoryNode::Size;
      newNode->current = newNode->begin;
      newNode->next = nullptr;

      assert(tail != nullptr);
      tail->next = newNode;
      tail = newNode;
    }
  public:
    MemoryManager() {
      head = new MemoryNode();
      head->begin = new std::byte[MemoryNode::Size];
      head->end = head->begin + MemoryNode::Size;
      head->current = head->begin;
      head->next = nullptr;
      tail = head;
    }

    ~MemoryManager() {
      MemoryNode* node = head;
      while (node) {
        delete[] node->begin;
        MemoryNode* nextNode = node->next;
        delete node;
        node = nextNode;
      }
      // overflows are allocated via std::aligned_alloc
      for (const auto& overflow : overflows)
        std::free(overflow.data);
      overflows.~vector();
    }

    bool isManaging(void* ptr) const {
      MemoryNode* node = head;
      while (node) {
        if (ptr >= node->begin && ptr < node->end)
          return true;
        node = node->next;
      }
      return false;
    }

    void* allocate(size_t size) {
      // overflow allocation
      if (size > MemoryNode::Size) {
        auto* overflow = static_cast<std::byte*>(std::malloc(size));
        overflows.emplace_back(overflow, size);
        return overflow;
      }

      // standard allocation
      auto* ptr = tail->current;
      if (ptr + size > tail->end) {
        allocateNewNode();
        ptr = tail->current;
      }
      tail->current = ptr + size;
      return ptr;
    }

    void* allocate(size_t size, size_t align) {
      // overflow allocation
      if (size + align > MemoryNode::Size) {
        auto* overflow = 
          static_cast<std::byte*>(std::aligned_alloc(align, size));
        overflows.emplace_back(overflow, size);
        return overflow;
      }

      const auto ensure_alignment = [align](std::byte* ptr) -> std::byte* {
        std::uintptr_t p = reinterpret_cast<std::uintptr_t>(ptr);
        p = (p + (align - 1)) & ~(align - 1);
        return reinterpret_cast<std::byte*>(p);
      };

      // standard allocation
      auto* ptr = ensure_alignment(tail->current);
      if (ptr + size > tail->end) {
        allocateNewNode();
        ptr = ensure_alignment(tail->current);
      }
      tail->current = ptr + size;
      assert(tail->current <= tail->end);
      return ptr;
    }

    size_t bytesAllocated() const {
      size_t total = 0;
      MemoryNode* node = head;
      while (node) {
        total += MemoryNode::Size;
        node = node->next;
      }
      for (const auto& overflow : overflows)
        total += overflow.size;
      return total;
    }

    size_t bytesAvailable() const {
      return tail->end - tail->current;
    }
  };

  MemoryManager memoryManager;
public:
  ASTContext() = default;

  ASTContext(const ASTContext&) = delete;
  ASTContext(ASTContext&&) = delete;
  ASTContext& operator=(const ASTContext&) = delete;
  ASTContext& operator=(ASTContext&&) = delete;

  ~ASTContext() = default;

  size_t bytesAllocated() const {
    return memoryManager.bytesAllocated();
  }

  size_t bytesAvailable() const {
    return memoryManager.bytesAvailable();
  }

  bool isManaging(void* ptr) const {
    return memoryManager.isManaging(ptr);
  }

  void* allocate(size_t size) {
    return memoryManager.allocate(size);
  }

  void* allocate(size_t size, size_t align) {
    return memoryManager.allocate(size, align);
  }

  ast::Identifier createIdentifier(const std::string& name) {
    return createIdentifier(std::string_view(name));
  }

  ast::Identifier createIdentifier(std::string_view name) {
    auto size = name.size();
    auto* ptr = memoryManager.allocate(size);
    std::memcpy(ptr, name.data(), size);
    return ast::Identifier(std::string_view(static_cast<char*>(ptr), size));
  }

  std::span<std::string_view> createSpan(
      std::string_view* begin, size_t size) {
    auto* ptr = memoryManager.allocate(size * sizeof(std::string_view));
    std::memcpy(ptr, begin, size * sizeof(std::string_view));
    return std::span<std::string_view>(static_cast<std::string_view*>(ptr),
                                       size);
  }

  template<typename T>
  std::span<T*> createSpan(T *const * begin, size_t size) {
    auto* ptr = memoryManager.allocate(size * sizeof(T*));
    std::memcpy(ptr, begin, size * sizeof(T*));
    return std::span<T*>(static_cast<T**>(ptr), size);
  }

  template<typename T>
  std::span<T*> createSpan(const std::vector<T*>& vec) {
    return createSpan(vec.data(), vec.size());
  }

  template<typename T, typename... Args>
  T* create(Args&&... args) {
    auto* ptr = memoryManager.allocate(sizeof(T), alignof(T));
    return new (ptr) T(std::forward<Args>(args)...);
  }
};

} // namespace ast
} // namespace cast::draft

inline void* operator new(size_t size, cast::draft::ast::ASTContext& ctx) {
  return ctx.allocate(size);
}

inline void* operator new(
    size_t size, std::align_val_t align, cast::draft::ast::ASTContext& ctx) {
  return ctx.allocate(size, static_cast<size_t>(align));
}

#endif // CAST_NEW_PARSER_ASTCONTEXT_H