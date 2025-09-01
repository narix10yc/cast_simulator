#include "utils/TaskDispatcher.h"

#include "utils/iocolor.h"
#include "utils/utils.h"
#include <condition_variable>
#include <iostream>
#include <mutex>

using namespace utils;

thread_local int TaskDispatcher::tls_local_version_ = 0;
thread_local TaskDispatcher::tls_instance_t
    TaskDispatcher::tls_local_instance_(nullptr, +[](void*) {});

TaskDispatcher::TaskDispatcher(int nWorkers)
    : tasks(), workers(), mtx(), cv(), syncCV(), nTotalTasks(0),
      nActiveWorkers(0), stopFlag(false) {
  assert(nWorkers > 0);
  workers.reserve(nWorkers);
  for (int i = 0; i < nWorkers; ++i)
    workers.emplace_back(&TaskDispatcher::worker_work, this);
}

void TaskDispatcher::enqueue(const std::function<void()>& task) {
  ++nTotalTasks;
  {
    std::lock_guard lock(mtx);
    if (stopFlag) {
      std::cerr << BOLDRED("[Err: ]")
                << "TaskDispatcher is stopped, cannot enqueue new tasks.\n";
      return;
    }
    tasks.push(std::move(task));
  }
  cv.notify_one();
}

void TaskDispatcher::ensure_tls_ready_() {
  std::lock_guard lock(tls_mutex_);
  if (tls_manager_.version == tls_local_version_) {
    // Already up to date
    return;
  }

  if (!tls_manager_.creator) {
    // No TLS installed
    tls_local_instance_.reset(nullptr);
    tls_local_version_ = 0;
    return;
  }

  tls_local_version_ = tls_manager_.version;
  tls_local_instance_ =
      tls_instance_t(tls_manager_.creator(), tls_manager_.deleter);
}

void TaskDispatcher::worker_work() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock lock(mtx);
      cv.wait(lock, [this]() { return stopFlag || !tasks.empty(); });

      if (stopFlag && tasks.empty()) {
        return;
      }

      if (tasks.empty()) {
        // No task to do. Keep waiting
        continue;
      }
      task = std::move(tasks.front());
      tasks.pop();
      ++nActiveWorkers;
    }
    ensure_tls_ready_();
    task();
    --nActiveWorkers;
    syncCV.notify_all();
  }
}

int TaskDispatcher::getWorkerID() const {
  // Everything read-only. No lock needed
  auto threadID = std::this_thread::get_id();
  for (int n = workers.size(), i = 0; i < n; ++i) {
    if (workers[i].get_id() == threadID)
      return i;
  }
  return -1;
}

void TaskDispatcher::sync(bool progressBar) {
  assert(getWorkerID() == -1 && "Cannot sync() inside a worker thread");
  {
    std::unique_lock lock(mtx);
    syncCV.wait(lock, [this, progressBar]() {
      if (progressBar) {
        int nTasksFinished = nTotalTasks - tasks.size() - nActiveWorkers;
        utils::displayProgressBar(nTasksFinished, nTotalTasks, 20);
      }
      return tasks.empty() && nActiveWorkers == 0;
    });
  }
  if (progressBar)
    std::cerr << std::endl;
  cv.notify_all();
}

void TaskDispatcher::join() {
  stopFlag = true;
  cv.notify_all();
  for (auto& thread : workers) {
    if (thread.joinable())
      thread.join();
  }
  // reset for reuse
  stopFlag = false;
  nTotalTasks = 0;
  nActiveWorkers = 0;
}