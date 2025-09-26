#ifndef UTILS_TASKDISPATCHER_H
#define UTILS_TASKDISPATCHER_H

#include <atomic>

#include <cassert>
#include <condition_variable>
#include <deque>
#include <functional>
#include <thread>

namespace utils {

/// Thread-safe task dispatcher
class TaskDispatcher {
  std::deque<std::function<void()>> tasks;
  std::vector<std::thread> workers;
  mutable std::mutex mtx;
  std::condition_variable cv;
  std::condition_variable syncCV;

  std::atomic<int> nTotalTasks = 0;
  std::atomic<int> nActiveWorkers = 0;
  std::atomic<bool> stopFlag = false;

  void worker_work();

public:
  TaskDispatcher(int nWorkers);

  TaskDispatcher(const TaskDispatcher&) = delete;
  TaskDispatcher(TaskDispatcher&&) = delete;

  TaskDispatcher& operator=(const TaskDispatcher&) = delete;
  TaskDispatcher& operator=(TaskDispatcher&&) = delete;

  ~TaskDispatcher() {
    assert(stopFlag == false);
    sync();
    stopFlag.store(true);
    cv.notify_all();
    for (auto& thread : workers)
      thread.join();
  }

  // Add a new task to the queue
  void enqueue(const std::function<void()>& task);

  int getNumWorkers() const { return workers.size(); }

  int getWorkerID() const;

  /// @brief A blocking method that waits until all tasks are finished.
  void sync(bool progressBar = false);

  /* TLS Management */
private:
  struct TLSManager {
    using creator_t = void* (*)();
    using deleter_t = void (*)(void*);

    int version = 0;
    creator_t creator = nullptr;
    deleter_t deleter = nullptr;
  };

  // Per-thread version and instance
  static thread_local int tls_local_version_;
  using tls_instance_t = std::unique_ptr<void, TLSManager::deleter_t>;
  static thread_local tls_instance_t tls_local_instance_;

  TLSManager tls_manager_; // single global manager state
  std::mutex tls_mutex_;

public:
  template <typename T> void installTLS() {
    std::lock_guard lock(tls_mutex_);
    tls_manager_.version++;
    tls_manager_.creator = +[]() -> void* { return new T(); };
    tls_manager_.deleter = +[](void* p) { delete static_cast<T*>(p); };
  }

  // Typed accessor inside tasks. Returns nullptr if no TLS installed or types
  // mismatch.
  template <class T> T* tls() noexcept {
    std::lock_guard lock(tls_mutex_);
    if (tls_local_version_ != tls_manager_.version)
      return nullptr;
    return static_cast<T*>(tls_local_instance_.get());
  }

  // Clear TLS. Increment version
  void clearTLS() {
    std::lock_guard lock(tls_mutex_);
    tls_manager_.version++;
    tls_manager_.creator = nullptr;
    tls_manager_.deleter = nullptr;
  }

private:
  // Every worker thread calls it before executing tasks
  void ensure_tls_ready_();

}; // TaskDispatcher

} // namespace utils

#endif // UTILS_TASKDISPATCHER_H
