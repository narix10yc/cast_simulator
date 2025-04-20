#include "gen_file.h"
#include "timeit/timeit.h"
#include "simulation/StatevectorCPU.h"
#include <iomanip>
#include <iostream>

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

// #include <immintrin.h>

int main(int argc, char* *argv) {

  int nthreads = 16;
  if (argc > 1)
    nthreads = std::stoi(argv[1]);

  Statevector sv(DEFAULT_nQubits);

  // sv.real[0] = 1.0;
  sv.randomize(nthreads);

  std::cerr << "Norm before: " << sv.normSquared(nthreads) << "\n";

#ifdef MULTI_THREAD_SIMULATION_KERNEL
  std::cerr << "Multi-threading enabled.\n";

  simulation_kernel(sv.real, sv.imag, DEFAULT_nQubits, nthreads);

  std::cerr << "Norm after: " << sv.normSquared(nthreads) << "\n";

#else
#endif

  return 0;
}