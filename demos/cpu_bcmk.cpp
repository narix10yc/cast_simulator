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

static double calculateGiBPerSecond(std::size_t memoryInBytes,
                                    double timeInSeconds) {
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

    // There is no factory method to create multi-qubit Hadamard gate,
    // so we construct it by matmul-ing multiple single-qubit Hadamard gates
    QuantumGatePtr hadamardGate = StandardQuantumGate::H(qubits[0]);
    for (int q = 1; q < k; ++q) {
      hadamardGate = cast::matmul(hadamardGate.get(),
                                  StandardQuantumGate::H(qubits[q]).get());
    }
    assert(hadamardGate->nQubits() == k);

    // While extremely unlikely, a randomly generated unitary gate could have
    // some entries whose absolute value is less than our zero-tolerance (which
    // is 1e-8 by default). We force to generate dense kernels here.
    kernelGenConfig.zeroTol = 0.0;
    kernelGenConfig.oneTol = 0.0;
    kernelMgr
        .genStandaloneGate(
            kernelGenConfig, unitaryGate, "unitary_gate_" + std::to_string(k))
        .consumeError(); // ignore possible error
    // And we relax forceDenseKernel for Hadamard gates
    kernelGenConfig.zeroTol = 1e-8;
    kernelGenConfig.oneTol = 1e-8;
    kernelMgr
        .genStandaloneGate(
            kernelGenConfig, hadamardGate, "hadamard_gate_" + std::to_string(k))
        .consumeError(); // ignore possible error
  }

  // Initialize JIT engine
  utils::timedExecute(
      [&]() {
        auto result =
            kernelMgr.initJIT(1, /* number of threads */
                              llvm::OptimizationLevel::O1, /* LLVM opt level */
                              false,                       /* use lazy JIT */
                              1 /* verbose (show progress bar) */
            );
        if (!result) {
          std::cerr << BOLDRED("[Error]: In initializing JIT engine: ")
                    << result.takeError() << "\n";
          std::exit(EXIT_FAILURE);
        }
      },
      "Initialize JIT Engine");

  // Create a statevector with NUM_QUBITS qubits
  cast::CPUStatevector<ScalarType> statevector(NUM_QUBITS, SIMD_WIDTH);
  utils::timedExecute([&]() { statevector.randomize(NUM_THREADS); },
                      "Initialize statevector");

  // Benchmark the kernels
  timeit::Timer timer(/* replication */ 7, /* verbose */ 0);
  timeit::TimingResult timingResult;
  for (int k = 1; k < 5; ++k) {
    std::cerr << "Benchmarking " << k << "-qubit unitary gate:  ";
    timingResult = timer.timeit([&]() {
      auto rst = kernelMgr.applyCPUKernel(statevector.data(),
                                          statevector.nQubits(),
                                          "unitary_gate_" + std::to_string(k),
                                          NUM_THREADS);
      if (!rst)
        std::cerr << BOLDRED("[Err]: ") << rst.takeError() << "\n";
    });
    std::cerr << timingResult.med * 1000 << " ms @ "
              << calculateGiBPerSecond(statevector.sizeInBytes(),
                                       timingResult.med)
              << " GiB/s\n";

    std::cerr << "Benchmarking " << k << "-qubit Hadamard gate: ";
    timingResult = timer.timeit([&]() {
      auto rst = kernelMgr.applyCPUKernel(statevector.data(),
                                          statevector.nQubits(),
                                          "hadamard_gate_" + std::to_string(k),
                                          NUM_THREADS);
      if (!rst)
        std::cerr << BOLDRED("[Err]: ") << rst.takeError() << "\n";
    });
    std::cerr << timingResult.med * 1000 << " ms @ "
              << calculateGiBPerSecond(statevector.sizeInBytes(),
                                       timingResult.med)
              << " GiB/s\n";
  }
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