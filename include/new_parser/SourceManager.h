#ifndef CAST_DRAFT_SOURCE_MANAGER_H
#define CAST_DRAFT_SOURCE_MANAGER_H

#include "new_parser/LocationSpan.h"
#include "utils/iocolor.h"
#include <iostream>
#include <vector>

namespace cast::draft {
namespace ast {

// SourceManager is responsible for managing the source code buffer
// and providing line information for error reporting and debugging.
// To load file, use loadFromFile or loadRawBuffer.
// Buffer is deleted in the destructor.
class SourceManager {
public:
  const char* bufferBegin;
  const char* bufferEnd;
  std::vector<const char*> lineTable;

  SourceManager() : bufferBegin(nullptr), bufferEnd(nullptr) {}

  SourceManager(const SourceManager&) = delete;
  SourceManager(SourceManager&&) = delete;
  SourceManager& operator=(const SourceManager&) = delete;
  SourceManager& operator=(SourceManager&&) = delete;

  ~SourceManager() { delete[] bufferBegin; }

  // return true on error
  bool loadFromFile(const char* filename);
  
  // return true on error
  bool loadRawBuffer(const char* buffer, size_t size);

  std::ostream& printLineInfo(std::ostream& os, LocationSpan loc) const;
};

} // namespace ast
} // namespace cast::draft

#endif // CAST_DRAFT_SOURCE_MANAGER_H