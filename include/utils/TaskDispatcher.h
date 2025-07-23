#ifndef UTILS_TASKDISPATCHER_H
#define UTILS_TASKDISPATCHER_H

#include <cassert>
#include <condition_variable>
#include <functional>
#include <queue>
#include <thread>

namespace utils {

/// Thread-safe task dispatcher
class TaskDispatcher {
  std::queue<std::function<void()>> tasks;
  std::vector<std::thread> workers;
  mutable std::mutex mtx;
  std::condition_variable cv;
  std::condition_variable syncCV;

  std::atomic<int> nTotalTasks;
  std::atomic<int> nActiveWorkers;
  std::atomic<bool> stopFlag;

  void workerThread();

public:
  TaskDispatcher(int nWorkers);

  TaskDispatcher(const TaskDispatcher&) = delete;
  TaskDispatcher(TaskDispatcher&&) = delete;

  TaskDispatcher& operator=(const TaskDispatcher&) = delete;
  TaskDispatcher& operator=(TaskDispatcher&&) = delete;

  ~TaskDispatcher() {
    if (stopFlag == false) {
      sync();
      join();
    }
  }

  // Add a new task to the queue
  void enqueue(const std::function<void()>& task);

  int getWorkerID() const;

  /// @brief A blocking method that waits until all tasks are finished.
  void sync(bool progressBar = false);

  /// @brief Join all threads. This method is automatically called upon
  /// destruction, so is normally not needed.
  void join();

}; // TaskDispatcher

} // namespace utils

#endif // UTILS_TASKDISPATCHER_H
