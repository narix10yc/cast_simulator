#ifndef CAST_DRAFT_SOURCE_MANAGER_H
#define CAST_DRAFT_SOURCE_MANAGER_H

#include "new_parser/LocationSpan.h"
#include <iostream>
#include <vector>
#include "utils/iocolor.h"

namespace cast::draft {

class SourceManager {
public:
  const char* bufferBegin;
  const char* bufferEnd;
  std::vector<const char*> lineTable;

  explicit SourceManager(const char* fileName);

  SourceManager(const SourceManager&) = delete;
  SourceManager(SourceManager&&) = delete;
  SourceManager& operator=(const SourceManager&) = delete;
  SourceManager& operator=(SourceManager&&) = delete;

  ~SourceManager() {
    delete[] bufferBegin;
  }

  std::ostream& printLineInfo(std::ostream& os, LocationSpan loc) const;
};


} // namespace cast::draft

#endif // CAST_DRAFT_SOURCE_MANAGER_H