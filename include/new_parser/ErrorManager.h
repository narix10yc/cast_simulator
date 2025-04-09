#ifndef CAST_DRAFT_ERROR_EMITTER_H
#define CAST_DRAFT_ERROR_EMITTER_H

#include "utils/iocolor.h"
#include <string>
#include <iostream>
#include <cassert>

namespace cast::draft {

class ErrorManager {
private:
  enum ErrorLevel {
    EL_Info,
    EL_Warning,
    EL_Error,
    EL_Fatal
  };
  struct ErrorItem {
    ErrorLevel level;
    std::string message;
    const char* lineBufferBegin;
    const char* lineBufferEnd;
    int lineNumber;
    int columnNumber;
  };

  int maxErrorCount;
  std::ostream& os;

}; // ErrorManager


} // namespace cast::draft


#endif // CAST_DRAFT_ERROR_EMITTER_H