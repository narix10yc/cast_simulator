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

static double computeGiBps(size_t memoryInBytes, double timeInSeconds) {
  return static_cast<double>(memoryInBytes) /
         (timeInSeconds * 1024 * 1024 * 1024);
}

inline std::ostream& loginfo() { return std::cout << BOLDCYAN("[Info]: "); }
inline std::ostream& logerr() { return std::cerr << BOLDRED("[Error]: "); }

template <typename ScalarType> static void benchmark() {
  static_assert(std::is_same_v<ScalarType, float> ||
                    std::is_same_v<ScalarType, double>,
                "ScalarType must be either float or double");

  CPUKernelManager km;
  CPUKernelGenConfig kernelGenConfig(
      SIMD_WIDTH,
      (std::is_same_v<ScalarType, float> ? Precision::FP32 : Precision::FP64));

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
    if (auto e = km.genGate(
            kernelGenConfig, unitaryGate, "u_gate_" + std::to_string(k))) {
      logerr() << "In kernel gen: " << llvm::toString(std::move(e)) << "\n";
      std::exit(EXIT_FAILURE);
    }
    // And we relax forceDenseKernel for Hadamard and RZ gates
    kernelGenConfig.zeroTol = 1e-8;
    kernelGenConfig.oneTol = 1e-8;
    if (auto e = km.genGate(
            kernelGenConfig, hadamardGate, "h_gate_" + std::to_string(k))) {
      logerr() << "In kernel gen: " << llvm::toString(std::move(e)) << "\n";
      std::exit(EXIT_FAILURE);
    }
    if (auto e = km.genGate(
            kernelGenConfig, sGate, "s_gate_" + std::to_string(k))) {
      logerr() << "In kernel gen: " << llvm::toString(std::move(e)) << "\n";
      std::exit(EXIT_FAILURE);
    }
  }

  // Initialize JIT engine
  utils::timedExecute(
      [&]() {
        if (auto e = km.compileAllPools(llvm::OptimizationLevel::O1)) {
          logerr() << "In initializing JIT engine: "
                   << llvm::toString(std::move(e)) << "\n";
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

  const auto run = [&](std::function<void()> f, const char* title) {
    auto tr = timer.timeit(f);
    std::cerr << "  " << title << ": " << computeGiBps(sv.sizeInBytes(), tr.min)
              << " GiB/s\n";
  };

  // clang-format off
  static constexpr auto denseTitle =    "Dense Gate    ";
  static constexpr auto hadamardTitle = "Hadamard Gate ";
  static constexpr auto sGateTitle =    "S Gate        ";
  // clang-format on

  for (int k = 1; k < 5; ++k) {
    std::cerr << "- " << k << "-qubit gates:\n";
    // Dense Gate
    run(
        [&]() {
          llvm::cantFail(km.applyCPUKernel(sv.data(),
                                           sv.nQubits(),
                                           "u_gate_" + std::to_string(k),
                                           NUM_THREADS));
        },
        denseTitle);

    // Hadamard Gate
    run(
        [&]() {
          llvm::cantFail(km.applyCPUKernel(sv.data(),
                                           sv.nQubits(),
                                           "h_gate_" + std::to_string(k),
                                           NUM_THREADS));
        },
        hadamardTitle);

    // S Gate
    run(
        [&]() {
          llvm::cantFail(km.applyCPUKernel(sv.data(),
                                           sv.nQubits(),
                                           "s_gate_" + std::to_string(k),
                                           NUM_THREADS));
        },
        sGateTitle);
  } // end for k
}

int main(int argc, char** argv) {
  loginfo() << "Benchmarking CPU simulation speed of multi-qubit dense gates "
            << "and multi-qubit Hadamard gates on CPU.\n";
  loginfo() << "Number of qubits  : " << NUM_QUBITS << "\n";
  loginfo() << "Number of threads : " << NUM_THREADS << "\n";
  loginfo() << "SIMD width        : " << static_cast<int>(SIMD_WIDTH)
            << " bits\n";

  loginfo() << "Starting single-precision test.\n";
  benchmark<float>();

  loginfo() << "Starting double-precision test.\n";
  benchmark<double>();
  return 0;
}