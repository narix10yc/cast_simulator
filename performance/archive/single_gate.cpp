#include "gen_file.h"
#include "timeit/timeit.h"
#include "simulation/StatevectorCPU.h"
#include <iomanip>
#include <iostream>

#ifdef USING_F32
using real_t = float;
#define REAL_T "f32"
#else
using real_t = double;
#define REAL_T "f64"
#endif
using namespace timeit;

#ifdef USING_ALT_KERNEL
using Statevector = utils::statevector::StatevectorAlt<real_t, S_VALUE>;
#else
using Statevector = utils::statevector::StatevectorSep<real_t>;
#endif

int main(int argc, char* *argv) {
  assert(argc > 1);
  unsigned targetQ = std::stoi(argv[1]);

  Statevector sv(DEFAULT_nQubits);
  Timer timer;
  // timer.setRunTime(0.5);
  timer.setReplication(7);
  TimingResult rst;

#ifdef MULTI_THREAD_SIMULATION_KERNEL
  std::cerr << "Multi-threading enabled.\n";

  // const std::vector<int> nthreads {2,4,8,12,16,20,24,28,32,36};
  // const std::vector<int> nthreads {16,24,32,36,48,64,68,72};
  const std::vector<int> nthreads{4, 8, 16, 24, 32};

  // const std::vector<int> nthreads {32, 64};

  std::vector<double> tarr(nthreads.size());

  int warmUpNThread = nthreads[nthreads.size() - 1];
  std::cerr << "Warm up run (" << warmUpNThread << "-thread):\n";
  rst = timer.timeit([&]() {
    simulation_kernel(sv.real, sv.imag, sv.nQubits, warmUpNThread);
  });
  rst.display();

  for (unsigned i = 0; i < nthreads.size(); i++) {
    int nthread = nthreads[i];
    std::cerr << nthread << "-thread:\n";
    rst = timer.timeit(
        [&]() { simulation_kernel(sv.real, sv.imag, sv.nQubits, nthread); });
    rst.display();
    tarr[i] = rst.min;
  }
  // easy to copy paste
  for (const auto& t : tarr)
    std::cerr << std::fixed << std::setprecision(4) << t << ",";
  std::cerr << "\n";

#else
  const uint64_t idxMax =
      1ULL << (DEFAULT_nQubits - S_VALUE - _metaData[targetQ].nQubits);

  rst = timer.timeit([&]() {
#ifdef USING_ALT_KERNEL
    _metaData[targetQ].func(sv.data, 0ULL, idxMax, _metaData[targetQ].mPtr);
#else
    _metaData[targetQ].func(sv.real, sv.imag, 0ULL, idxMax,
                            _metaData[targetQ].mPtr);
#endif
  });
  rst.display();
#endif

  return 0;
}