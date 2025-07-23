#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "utils/iocolor.h"

namespace utils {

class Logger {
private:
  std::shared_ptr<std::ofstream> fileStream;
  std::ostream* outStream;

public:
  int verbosity;

  Logger(std::nullptr_t) : verbosity(0), outStream(nullptr) {}

  Logger(std::ostream& stream, int verbosity = 1)
      : verbosity(verbosity), outStream(&stream) {}

  // Construct from file path
  Logger(const char* filePath, int verbosity = 1)
      : verbosity(verbosity),
        fileStream(std::make_unique<std::ofstream>(filePath, std::ios::app)),
        outStream(fileStream.get()) {
    if (!fileStream->is_open()) {
      std::cerr << BOLDYELLOW("Warning: ")
                << "Logger: Could not open log file '" << filePath
                << "'. Fall back to stderr.\n";
      outStream = &std::cerr;
    }
  }

  // Stream-like object returned by log()
  class LogStream {
  private:
    std::ostream* stream_;

  public:
    LogStream(std::ostream* stream) : stream_(stream) {}

    template <typename T> LogStream& operator<<(const T& value) {
      if (stream_)
        (*stream_) << value;
      return *this;
    }

    // Support manipulators like std::endl
    LogStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
      if (stream_)
        (*stream_) << manip;
      return *this;
    }
  };

  // Get a log stream for the given level
  LogStream log(int level) {
    return (level <= verbosity) ? LogStream(outStream) : LogStream(nullptr);
  }
};

} // namespace utils