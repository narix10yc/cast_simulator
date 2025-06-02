/*
  cpu_bcmk.cpp
  A demo for benchmarking the simulation speed of multi-qubit dense gates and
  multi-qubit Hadamard gates on CPU.
*/

// CPU/CPUKernelManager.h contains all components needed for CPU simulation
#include "cast/CPU/CPUKernelManager.h"

// CPUStatevector class
#include "cast/CPU/CPUStatevector.h"

// Timing utilities
#include "timeit/timeit.h"

// For sampling qubits
#include "utils/utils.h"

using namespace cast;

// 28-qubit statevector takes 2GiB memory for float and 4GiB for double
constexpr int NUM_QUBITS = 28;

// Number of threads to use for benchmarking.
constexpr int NUM_THREADS = 10;

// Unit in bytes.
// 16 for ARM_NEON and SSE
// 32 for AVX2
// 64 for AVX512
constexpr int SIMD_REG_SIZE = 16;

static_assert(SIMD_REG_SIZE == 16 ||
              SIMD_REG_SIZE == 32 ||
              SIMD_REG_SIZE == 64);

constexpr int GetSimdS_F32() {
  if constexpr (SIMD_REG_SIZE == 16)
    return 2; // <4 x float>
  if constexpr (SIMD_REG_SIZE == 32)
    return 3; // <8 x float>
  if constexpr (SIMD_REG_SIZE == 64)
    return 4; // <16 x float>
  return 0; // Unreachable
}

constexpr int GetSimdS_F64() {
  if constexpr (SIMD_REG_SIZE == 16)
    return 1; // <2 x double>
  if constexpr (SIMD_REG_SIZE == 32)
    return 2; // <4 x double>
  if constexpr (SIMD_REG_SIZE == 64)
    return 3; // <8 x double>
  return 0; // Unreachable
}

constexpr int SIMD_S_F32 = GetSimdS_F32();
constexpr int SIMD_S_F64 = GetSimdS_F64();

template<typename ScalarType>
static void benchmark() {
  static_assert(std::is_same_v<ScalarType, float> ||
                std::is_same_v<ScalarType, double>,
                "ScalarType must be either float or double");
  constexpr int SIMD_S = std::is_same_v<ScalarType, float> ? SIMD_S_F32
                                                           : SIMD_S_F64;

  CPUKernelManager kernelMgr;
  CPUKernelGenConfig kernelGenConfig;
  kernelGenConfig.simd_s = SIMD_S;
  kernelGenConfig.precision = std::is_same_v<ScalarType, float> ? 32 : 64;

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
      hadamardGate = cast::matmul(
        hadamardGate.get(), StandardQuantumGate::H(qubits[q]).get());
    }

    // While extremely unlikely, a randomly generated unitary gate could have 
    // some entries whose absolute value is less than our zero-tolerance (which
    // is 1e-8 by default). We set forceDenseKernel to always generate 
    // dense-gate kernels.
    kernelGenConfig.forceDenseKernel = true;
    kernelMgr.genCPUGate(kernelGenConfig, unitaryGate,
                         "unitary_gate_" + std::to_string(k));
    // And we relax forceDenseKernel for Hadamard gates
    kernelGenConfig.forceDenseKernel = false;
    kernelMgr.genCPUGate(kernelGenConfig, hadamardGate,
                         "hadamard_gate_" + std::to_string(k));
  }

  // Initialize JIT engine
  utils::timedExecute([&]() {
    kernelMgr.initJIT(
      1,  /* number of threads */
      llvm::OptimizationLevel::O1, /* LLVM opt level */
      false, /* use lazy JIT */
      1 /* verbose (show progress bar) */
    );
  }, "Initialize JIT Engine");

  // Create a statevector with NUM_QUBITS qubits
  cast::CPUStatevector<ScalarType> statevector(NUM_QUBITS, SIMD_S);
  utils::timedExecute([&]() {
    statevector.randomize(1); // Randomize the statevector with 1 thread
  }, "Initialize statevector");

  // Benchmark the kernels
  timeit::Timer timer(/* replication */ 3, /* verbose */ 0);
  timeit::TimingResult timingResult;
  for (int k = 1; k < 5; ++k) {
    std::cerr << "Benchmarking " << k << "-qubit unitary gate:\n";
    timingResult = timer.timeit([&]() {
      kernelMgr.applyCPUKernelMultithread(
        statevector.data(),
        statevector.nQubits(), 
        "unitary_gate_" + std::to_string(k),
        NUM_THREADS
      );
    });
    timingResult.display(/* num significant digits */ 4, std::cerr) << "\n";

    std::cerr << "Benchmarking " << k << "-qubit Hadamard gate:\n";
    timingResult = timer.timeit([&]() {
      kernelMgr.applyCPUKernelMultithread(
        statevector.data(),
        statevector.nQubits(), 
        "hadamard_gate" + std::to_string(k),
        NUM_THREADS
      );
    });
    timingResult.display(/* num significant digits */ 4, std::cerr) << "\n";
  }
}

int main(int argc, char** argv) {
  benchmark<float>();
  benchmark<double>();
  return 0;
}