#include "cast/_config.h"
#include "utils/iocolor.h"

#include <thread>
#include <cassert>
#include <cstdlib>
#include <iostream>

using namespace cast;

static int parsePositiveInt(const char* str) {
  if (str == nullptr)
    return 0;

  int value = 0;
  while (*str) {
    if (!std::isdigit(*str))
      return 0;
    value = value * 10 + (*str - '0');
    ++str;
  }
  return value;
}

int cast::get_num_threads() {
  const char* env = std::getenv("CAST_NUM_THREADS");
  int envNThreads = parsePositiveInt(env);
  if (envNThreads > 0)
    return envNThreads;

  // CMAKE_CAST_NUM_THREADS is forwarded from CMake by CMake option
  // -DCAST_NUM_THREADS=<N>
  #ifdef CMAKE_CAST_NUM_THREADS
  if (CMAKE_CAST_NUM_THREADS > 0)
    return CMAKE_CAST_NUM_THREADS;
  #endif // #ifdef CMAKE_CAST_NUM_THREADS

  int nThreads = std::thread::hardware_concurrency();
  if (nThreads <= 0) {
    std::cerr << BOLDYELLOW("[Warning]: ")
      "Unable to detect hardware concurrency, defaulting to 1 thread. "
      "You can set the number of threads using the CAST_NUM_THREADS "
      "environment variable or by defining -DCAST_NUM_THREADS=<N> "
      "when configuring the project.\n";
    nThreads = 1;
  }
  return nThreads;
}