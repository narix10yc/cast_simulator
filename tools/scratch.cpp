#include "utils/MaybeError.h"
#include "utils/TaskDispatcher.h"
#include <iostream>

using namespace cast;

int main(int argc, char** argv) {

  utils::TaskDispatcher dispatcher(4);
  std::mutex mtx;

  struct TLS {
    int id;
  };

  dispatcher.installTLS<TLS>();

  for (int i = 0; i < 10; ++i) {
    dispatcher.enqueue([&, i]() {
      if (auto* tls = dispatcher.tls<TLS>()) {
        std::lock_guard lock(mtx);
        tls->id++;
        std::cout << "TLS ID for thread " << dispatcher.getWorkerID() << ": "
                  << tls->id << "\n";
      }
      std::lock_guard lock(mtx);
      std::cout << "Processing task " << i << " on thread "
                << dispatcher.getWorkerID() << "\n";
    });
  }

  dispatcher.sync();

  return 0;
}