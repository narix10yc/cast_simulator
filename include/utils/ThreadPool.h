#ifndef UTILS_THREADPOOL_H
#define UTILS_THREADPOOL_H

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include <llvm/Support/Error.h>

#include "utils/utils.h"

namespace utils {

// An empty struct
struct NoTLS {};

/// A simple thread pool implementation.
/// @tparam TLS The type of thread-local storage. Use `NoTLS` for no TLS.
/// @note The thread-local storage instance will be default-constructed for
/// each worker thread. TLS can be accessed inside tasks via the static method
/// `ThreadPool::tls()`.
template <typename TLS = NoTLS> class ThreadPool {
  static constexpr bool hasTLS = !std::is_same_v<TLS, NoTLS>;

  std::deque<std::function<void()>> tasks;
  std::vector<std::thread> workers;
  mutable std::mutex mtx;
  std::condition_variable cv;
  std::condition_variable syncCV;
  llvm::Error err = llvm::Error::success();

  std::atomic<int> nTotalTasks = 0;
  std::atomic<int> nActiveWorkers = 0;
  std::atomic<bool> stopFlag = false;

  void worker_work_() {
    std::function<void()> task;
    bool tlExit = false;
    while (true) {
      std::unique_lock lk(mtx);
      cv.wait(lk, [this, &tlExit]() {
        tlExit = stopFlag.load();
        return tlExit || !tasks.empty();
      });

      if (tlExit)
        break;

      assert(!tasks.empty());
      task = std::move(tasks.front());
      tasks.pop_front();

      lk.unlock();
      ++nActiveWorkers;
      task();
      --nActiveWorkers;
      lk.lock();

      // we always notify the syncing threads for it to update progress bar
      syncCV.notify_all();
    }
  }

public:
  ThreadPool(int nWorkers) {
    assert(nWorkers > 0);
    workers.reserve(nWorkers);
    for (int i = 0; i < nWorkers; ++i)
      workers.emplace_back(&ThreadPool::worker_work_, this);
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;

  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  ~ThreadPool() {
    assert(stopFlag == false);
    sync();
    stopFlag.store(true);
    cv.notify_all();
    for (auto& thread : workers)
      thread.join();
    assert(!err);
    llvm::consumeError(std::move(err));
  }

  // Add a new task to the queue. If the task may err (via llvm::Error), use
  // `enqueueMayErr()` instead.
  void enqueue(const std::function<void()>& task) {
    nTotalTasks.store(nTotalTasks.load() + 1);
    {
      std::lock_guard lock(mtx);
      assert(stopFlag == false &&
             "Cannot enqueue tasks after join() or during destruction");
      tasks.push_back(std::move(task));
    }
    cv.notify_one();
  }

  // Enqueue a task that may return an error (via llvm::Error). If the function
  // does err, the error will be concatenated internally and can be retrieved
  // by `this->takeError()`.
  void enqueueMayErr(const std::function<llvm::Error()>& task) {
    enqueue([this, task]() {
      if (auto e = task()) {
        std::lock_guard lk(mtx);
        this->err = llvm::joinErrors(std::move(this->err), std::move(e));
      }
    });
  }

  bool hasError() const {
    std::lock_guard lk(mtx);
    // llvm::Error::operator bool() is non-const
    return static_cast<bool>(const_cast<llvm::Error&>(err));
  }

  llvm::Error takeError() {
    std::lock_guard lk(mtx);
    return std::exchange(err, llvm::Error::success());
  }

  int getNumWorkers() const { return workers.size(); }

  int getWorkerID() const {
    // Everything read-only. No lock needed
    auto threadID = std::this_thread::get_id();
    for (int n = workers.size(), i = 0; i < n; ++i) {
      if (workers[i].get_id() == threadID)
        return i;
    }
    return -1;
  }

  bool isIdle() const {
    return nTotalTasks.load() == 0 && nActiveWorkers.load() == 0;
  }

  /// @brief A blocking method that waits until all tasks are finished.
  void sync(bool progressBar = false) {
    assert(getWorkerID() == -1 && "Cannot sync() inside a worker thread");
    {
      std::unique_lock lk(mtx);
      syncCV.wait(lk, [this, progressBar]() {
        if (progressBar) {
          int nTasksFinished = nTotalTasks - tasks.size() - nActiveWorkers;
          utils::displayProgressBar(nTasksFinished, nTotalTasks, 20);
        }
        return tasks.empty() && nActiveWorkers == 0;
      });
    }
    if (progressBar)
      std::cerr << "\n" << std::flush;

    // reset for reuse
    nTotalTasks = 0;
  }

  static TLS& tls() {
    if constexpr (std::is_void_v<TLS>)
      return nullptr;
    return tls_local_instance_;
  }

private:
  static thread_local TLS tls_local_instance_;

}; // TaskDispatcher

// Default ctor
template <typename TLS> thread_local TLS ThreadPool<TLS>::tls_local_instance_{};

} // namespace utils

#endif // UTILS_THREADPOOL_H
