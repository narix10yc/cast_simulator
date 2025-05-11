#include "new_parser/SourceManager.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace cast::draft;

SourceManager::SourceManager(const char* fileName) {
  std::ifstream file(fileName, std::ifstream::binary);
  assert(file);
  assert(file.is_open());

  file.seekg(0, file.end);
  auto l = file.tellg();
  file.seekg(0, file.beg);

  bufferBegin = new char[l];
  bufferEnd = bufferBegin + l;
  file.read(const_cast<char*>(bufferBegin), l);
  file.close();
  
  // Initialize line table
  lineTable.push_back(bufferBegin);
  for (const char* ptr = bufferBegin; ptr < bufferEnd; ++ptr) {
    if (*ptr == '\n')
      lineTable.push_back(ptr + 1);
  }
  lineTable.push_back(bufferEnd);
}

std::ostream& SourceManager::printLineInfo(
    std::ostream& os, LocationSpan loc) const {
  auto findLine = [this](const char* ptr) {
    return --std::upper_bound(lineTable.begin(), lineTable.end(), ptr);
  };
  assert(!lineTable.empty());
  assert(loc.begin >= lineTable[0]);
  assert(loc.end <= lineTable.back());
  auto beginLineIt = findLine(loc.begin);
  auto endLineIt = findLine(loc.end);

  int beginLineNumber = beginLineIt - lineTable.begin() + 1;
  int endLineNumber = endLineIt - lineTable.begin() + 1;
  int beginColNumber = loc.begin - *beginLineIt;
  int endColNumber = loc.end - *endLineIt;
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