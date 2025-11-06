#include "utils/TaskDispatcher.h"

#include "utils/utils.h"
#include <condition_variable>
#include <iostream>
#include <mutex>

using namespace utils;

thread_local int TaskDispatcher::tls_local_version_ = 0;
thread_local TaskDispatcher::tls_instance_t
    TaskDispatcher::tls_local_instance_(nullptr, +[](void*) {});

TaskDispatcher::TaskDispatcher(int nWorkers) {
  assert(nWorkers > 0);
  workers.reserve(nWorkers);
  for (int i = 0; i < nWorkers; ++i)
    workers.emplace_back(&TaskDispatcher::worker_work, this);
}

void TaskDispatcher::enqueue(const std::function<void()>& task) {
  nTotalTasks.store(nTotalTasks.load() + 1);
  {
    std::lock_guard lock(mtx);
    assert(stopFlag == false &&
           "Cannot enqueue tasks after join() or during destruction");
    tasks.push_back(std::move(task));
  }
  cv.notify_one();
}

void TaskDispatcher::ensure_tls_ready_() {
  if (tls_manager_.version == tls_local_version_) {
    // Already up to date
    return;
  }

  if (tls_manager_.creator == nullptr) {
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
    ensure_tls_ready_();
    task();
    --nActiveWorkers;
    lk.lock();

    // we always notify the syncing threads for it to update progress bar
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
