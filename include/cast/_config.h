#ifndef CAST_CONFIG_H
#define CAST_CONFIG_H

namespace cast {
  // Get the number of threads. Priority order:
  // 1. Environment variable CAST_NUM_THREADS
  // 2. -DCAST_NUM_THREADS=<N> during configuration
  // 3. Hardware concurrency (std::thread::hardware_concurrency())
  // 4. Default to 1 thread if all else fails
  int get_num_threads();
}

#endif // CAST_CONFIG_H