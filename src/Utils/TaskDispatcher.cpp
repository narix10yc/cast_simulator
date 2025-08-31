#include "utils/TaskDispatcher.h"

#include "utils/iocolor.h"
#include "utils/utils.h"
#include <condition_variable>
#include <iostream>
#include <mutex>

using namespace utils;

TaskDispatcher::TaskDispatcher(int nWorkers)
    : tasks(), workers(), mtx(), cv(), syncCV(), nTotalTasks(0),
      nActiveWorkers(0), stopFlag(false) {
  workers.reserve(nWorkers);
  for (int i = 0; i < nWorkers; ++i) {
    workers.emplace_back([this]() { workerThread(); });
  }
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

void TaskDispatcher::workerThread() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock lock(mtx);
      cv.wait(lock, [this]() { return stopFlag || !tasks.empty(); });

      if (stopFlag && tasks.empty()) {
        return;
      }

      if (!tasks.empty()) {
        task = std::move(tasks.front());
        tasks.pop();
      } else {
        continue;
      }
      ++nActiveWorkers;
    }
    task();
    --nActiveWorkers;
    syncCV.notify_one();
  }
}

int TaskDispatcher::getWorkerID() const {
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