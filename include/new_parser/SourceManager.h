#ifndef CAST_DRAFT_SOURCE_MANAGER_H
#define CAST_DRAFT_SOURCE_MANAGER_H

#include <fstream>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "utils/iocolor.h"

namespace cast::draft {

class SourceManager {
public:
  const char* bufferBegin;
  const char* bufferEnd;
  std::vector<const char*> lineTable;

  explicit SourceManager(const char* fileName) {
    std::ifstream file(fileName, std::ifstream::binary);
    assert(file);
    assert(file.is_open());
  
    file.seekg(0, file.end);
    auto bufferLength = file.tellg();
    file.seekg(0, file.beg);
  
    bufferBegin = new char[bufferLength];
    bufferEnd = bufferBegin + bufferLength;
    file.read(const_cast<char*>(bufferBegin), bufferLength);
    file.close();
    
    // Initialize line table
    lineTable.push_back(bufferBegin);
    for (const char* ptr = bufferBegin; ptr < bufferEnd; ++ptr) {
      if (*ptr == '\n') {
        lineTable.push_back(ptr + 1);
      }
    }
    lineTable.push_back(bufferEnd);
  }

  SourceManager(const SourceManager&) = delete;
  SourceManager(SourceManager&&) = delete;
  SourceManager& operator=(const SourceManager&) = delete;
  SourceManager& operator=(SourceManager&&) = delete;

  ~SourceManager() = default;

  std::ostream& printLineInfo(
      std::ostream& os, const char* bufferBegin, const char* bufferEnd) const {
    
    auto findLine = [this](const char* ptr) {
      return --std::upper_bound(lineTable.begin(), lineTable.end(), ptr);
    };
    
    assert(bufferBegin < bufferEnd);
    auto beginLineIt = findLine(bufferBegin);
    auto endLineIt = findLine(bufferEnd);
    assert(beginLineIt != lineTable.end());
    assert(endLineIt != lineTable.end());

    int beginLineNumber = beginLineIt - lineTable.begin() + 1;
    int endLineNumber = endLineIt - lineTable.begin() + 1;
    int beginColNumber = bufferBegin - *beginLineIt;
    int endColNumber = bufferEnd - *beginLineIt;
    assert(beginColNumber >= 0);
    assert(endColNumber >= 0);

    if (beginLineNumber != endLineNumber) {
      return os << "Lines spanning "
                << beginLineNumber << ":" << beginColNumber
                << " to "
                << endLineNumber << ":" << endColNumber << "\n";
    }

    os << beginLineNumber << ": ";
    os.write(*beginLineIt, *(std::next(beginLineIt)) - *beginLineIt);
    os << std::string(beginColNumber + std::log10(beginLineNumber) + 3, ' ')
       << BOLDGREEN(std::string(endColNumber - beginColNumber, '^'))
       << "\n";

    os << "\n";
    return os;
  }

};


} // namespace cast::draft

#endif // CAST_DRAFT_SOURCE_MANAGER_H