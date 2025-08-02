/*
  cpu_bcmk.cpp
  A demo for benchmarking the simulation speed of multi-qubit dense gates and
  multi-qubit Hadamard gates on CPU.
  Its executable takes no arguments.
*/

// CPU/CPUKernelManager.h contains all components needed for CPU simulation
#include "cast/CPU/CPUKernelManager.h"

// CPUStatevector class
#include "cast/CPU/CPUStatevector.h"

// Timing utilities
#include "timeit/timeit.h"

using namespace cast;

// 28-qubit statevector takes 2GiB memory for float and 4GiB for double
constexpr int NUM_QUBITS = 28;

// Number of threads to use for benchmarking.
int NUM_THREADS = cast::get_cpu_num_threads();
CPUSimdWidth SIMD_WIDTH = cast::get_cpu_simd_width();

static double computeGiBps(std::size_t memoryInBytes, double timeInSeconds) {
  return static_cast<double>(memoryInBytes) /
         (timeInSeconds * 1024 * 1024 * 1024);
}

template <typename ScalarType> static void benchmark() {
  static_assert(std::is_same_v<ScalarType, float> ||
                    std::is_same_v<ScalarType, double>,
                "ScalarType must be either float or double");

  CPUKernelManager kernelMgr;
  CPUKernelGenConfig kernelGenConfig(
      SIMD_WIDTH,
      (std::is_same_v<ScalarType, float> ? Precision::F32 : Precision::F64));

  // Generate 1 to 4-qubit random unitary and Hadamard gates
  for (int k = 1; k < 5; ++k) {
    std::vector<int> qubits;
    utils::sampleNoReplacement(NUM_QUBITS, k, qubits);

    // Random multi-qubit unitary gates are easy to construct
    auto unitaryGate = StandardQuantumGate::RandomUnitary(qubits);

    // There is no factory method to create multi-qubit Hadamard or RZ gate,
    // so we construct it by matmul-ing multiple single-qubit ones
    QuantumGatePtr hadamardGate = StandardQuantumGate::H(qubits[0]);
    for (int q = 1; q < k; ++q) {
      hadamardGate = cast::matmul(hadamardGate.get(),
                                  StandardQuantumGate::H(qubits[q]).get());
    }
    assert(hadamardGate->nQubits() == k);

    QuantumGatePtr sGate = StandardQuantumGate::S(qubits[0]);
    for (int q = 1; q < k; ++q) {
      sGate =
          cast::matmul(sGate.get(), StandardQuantumGate::S(qubits[q]).get());
    }

    // While extremely unlikely, a randomly generated unitary gate could have
    // some entries whose absolute value is less than our zero-tolerance (which
    // is 1e-8 by default). We force to generate dense kernels here.
    kernelGenConfig.zeroTol = 0.0;
    kernelGenConfig.oneTol = 0.0;
    kernelMgr
        .genStandaloneGate(
            kernelGenConfig, unitaryGate, "u_gate_" + std::to_string(k))
        .consumeError(); // ignore possible error
    // And we relax forceDenseKernel for Hadamard and RZ gates
    kernelGenConfig.zeroTol = 1e-8;
    kernelGenConfig.oneTol = 1e-8;
    kernelMgr
        .genStandaloneGate(
            kernelGenConfig, hadamardGate, "h_gate_" + std::to_string(k))
        .consumeError(); // ignore possible error
    kernelMgr
        .genStandaloneGate(kernelGenConfig,
                           sGate,
                           "s_gate_" + std::to_string(k))
        .consumeError(); // ignore possible error
  }

  // Initialize JIT engine
  utils::timedExecute(
      [&]() {
        auto r =
            kernelMgr.initJIT(1, /* number of threads */
                              llvm::OptimizationLevel::O1, /* LLVM opt level */
                              false,                       /* use lazy JIT */
                              1 /* verbose (show progress bar) */
            );
        if (!r) {
          std::cerr << BOLDRED("[Error]: ")
                    << "In initializing JIT engine: " << r.takeError() << "\n";
          std::exit(EXIT_FAILURE);
        }
      },
      "Initialize JIT Engine");

  // Create a statevector with NUM_QUBITS qubits
  cast::CPUStatevector<ScalarType> sv(NUM_QUBITS, SIMD_WIDTH);
  utils::timedExecute([&]() { sv.randomize(NUM_THREADS); },
                      "Initialize statevector");

  // Benchmark the kernels
  timeit::Timer timer(/* replication */ 7, /* verbose */ 0);
  timeit::TimingResult tr;
  for (int k = 1; k < 5; ++k) {
    std::cerr << "- " << k << "-qubit gates:\n";
    // Dense Gate
    tr = timer.timeit([&]() {
      kernelMgr
          .applyCPUKernel(sv.data(),
                          sv.nQubits(),
                          "u_gate_" + std::to_string(k),
                          NUM_THREADS)
          .consumeError();
    });
    std::cerr << "  Dense Gate: " << computeGiBps(sv.sizeInBytes(), tr.min)
              << " GiB/s\n";

    // Hadamard Gate
    tr = timer.timeit([&]() {
      kernelMgr
          .applyCPUKernel(sv.data(),
                          sv.nQubits(),
                          "h_gate_" + std::to_string(k),
                          NUM_THREADS)
          .consumeError();
    });
    std::cerr << "  H Gate:     " << computeGiBps(sv.sizeInBytes(), tr.min)
              << " GiB/s\n";

    // S Gate
    tr = timer.timeit([&]() {
      kernelMgr
          .applyCPUKernel(sv.data(),
                          sv.nQubits(),
                          "s_gate_" + std::to_string(k),
                          NUM_THREADS)
          .consumeError();
    });
    std::cerr << "  S Gate:     " << computeGiBps(sv.sizeInBytes(), tr.min)
              << " GiB/s\n";
  } // end for k
}

int main(int argc, char** argv) {
  std::cerr << BOLDCYAN("[Info]: ")
            << "Benchmarking CPU simulation speed of multi-qubit dense gates "
            << "and multi-qubit Hadamard gates on CPU.\n";
  std::cerr << BOLDCYAN("[Info]: ") << "Using " << NUM_QUBITS
            << "-qubit statevector with " << NUM_THREADS << " threads.\n";
  std::cerr << BOLDCYAN("[Info]: ") << "SIMD width is set to "
            << static_cast<int>(SIMD_WIDTH) << " bits.\n";

  std::cerr << BOLDCYAN("[Info]: ") << "Starting single-precision test.\n";
  benchmark<float>();

  std::cerr << BOLDCYAN("[Info]: ") << "Starting double-precision test.\n";
  benchmark<double>();
  return 0;
}