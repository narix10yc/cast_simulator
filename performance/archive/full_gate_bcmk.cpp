#include "gen_file.h"
#include "timeit/timeit.h"
#include "simulation/StatevectorCPU.h"
#include <iomanip>
#include <iostream>
#include <string>

#ifdef USING_F32
using Statevector = utils::statevector::StatevectorSep<float>;
using real_t = float;
#define REAL_T "f32"
#else
using Statevector = utils::statevector::StatevectorSep<double>;
using real_t = double;
#define REAL_T "f64"
#endif
using namespace timeit;

int main(int argc, char* *argv) {
  assert(argc > 1);
  const std::string test_name = argv[1];

  real_t* real, *imag;
  Timer timer;
  timer.setRunTime(0.5);
  // timer.setReplication(3);
  TimingResult rst;

#ifdef MULTI_THREAD_SIMULATION_KERNEL
#else
  std::cerr << "\n============ New Run ============\n";
  int count = 0;
  for (unsigned nQubits : {8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28}) {
    // for (const int nQubits : { DEFAULT_nQubits, DEFAULT_nQubits }) {
    if (nQubits < 20)
      timer.setReplication(7);
    else
      timer.setReplication(3);
    uint64_t idxMax = 1ULL << (nQubits - SIMD_S - _metaData[0].nQubits);

    real = (real_t*)std::aligned_alloc(64,
                                        2 * (1ULL << nQubits) * sizeof(real_t));
    imag = real + (1ULL << nQubits);

    rst = timer.timeit([&]() {
      for (unsigned i = count; i < count + nQubits; ++i) {
#ifdef USING_ALT_KERNEL
        _metaData[i].func(real, 0, idxMax, _metaData[i].mPtr);
#else
        _metaData[i].func(real, imag, 0, idxMax, _metaData[i].mPtr);
#endif
      }
    });
    count += nQubits;
    // rst.display();
    std::free(real);

    std::cerr << "ours-alt," << test_name << "," << nQubits << "," REAL_T ","
              << std::scientific << std::setprecision(4) << (rst.min / nQubits)
              << "\n";
  }
#endif

  return 0;
}