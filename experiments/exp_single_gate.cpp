// CPU/CPUKernelManager.h contains all components needed for CPU simulation
#include "cast/CPU/CPUKernelManager.h"

// CPUStatevector class
#include "cast/CPU/CPUStatevector.h"

// Timing utilities
#include "timeit/timeit.h"

using namespace cast;

enum GateType { U1, H1, S1, U2, H2, S2, U3, H3, S3, U4, H4, S4 };

static std::ostream& operator<<(std::ostream& os, GateType gateType) {
  switch (gateType) {
  case U1:
    return os << "u1";
  case H1:
    return os << "h1";
  case S1:
    return os << "s1";
  case U2:
    return os << "u2";
  case H2:
    return os << "h2";
  case S2:
    return os << "s2";
  case U3:
    return os << "u3";
  case H3:
    return os << "h3";
  case S3:
    return os << "s3";
  case U4:
    return os << "u4";
  case H4:
    return os << "h4";
  case S4:
    return os << "s4";
  default:
    return os << "??";
  }
}

// 28-qubit statevector takes 2GiB memory for float and 4GiB for double
static constexpr int NUM_QUBITS = 28;

static void createGates(std::vector<QuantumGatePtr>& gates, GateType gateType) {
  switch (gateType) {
  case U1: {
    for (int q = 0; q < NUM_QUBITS; ++q)
      gates.push_back(StandardQuantumGate::RandomUnitary(q));
    break;
  }
  case H1: {
    for (int q = 0; q < NUM_QUBITS; ++q)
      gates.push_back(StandardQuantumGate::H(q));
    break;
  }
  case S1: {
    for (int q = 0; q < NUM_QUBITS; ++q)
      gates.push_back(StandardQuantumGate::S(q));
    break;
  }
  case U2: {
    for (int q = 0; q < NUM_QUBITS; ++q) {
      gates.push_back(
          StandardQuantumGate::RandomUnitary({q, (q + 1) % NUM_QUBITS}));
    }
    break;
  }
  case H2: {
    for (int q = 0; q < NUM_QUBITS; ++q) {
      QuantumGatePtr gate = StandardQuantumGate::H(q);
      gate = cast::matmul(gate.get(),
                          StandardQuantumGate::H((q + 1) % NUM_QUBITS).get());
      gates.push_back(gate);
    }
    break;
  }
  case S2: {
    for (int q = 0; q < NUM_QUBITS; ++q) {
      QuantumGatePtr gate = StandardQuantumGate::S(q);
      gate = cast::matmul(gate.get(),
                          StandardQuantumGate::S((q + 1) % NUM_QUBITS).get());
      gates.push_back(gate);
    }
    break;
  }
  case U3: {
    for (int q = 0; q < NUM_QUBITS; ++q)
      gates.push_back(StandardQuantumGate::RandomUnitary(
          {q, (q + 1) % NUM_QUBITS, (q + 2) % NUM_QUBITS}));
    break;
  }
  case H3: {
    for (int q = 0; q < NUM_QUBITS; ++q) {
      QuantumGatePtr gate = StandardQuantumGate::H(q);
      gate = cast::matmul(gate.get(),
                          StandardQuantumGate::H((q + 1) % NUM_QUBITS).get());
      gate = cast::matmul(gate.get(),
                          StandardQuantumGate::H((q + 2) % NUM_QUBITS).get());
      gates.push_back(gate);
    }
    break;
  }
  case S3: {
    for (int q = 0; q < NUM_QUBITS; ++q) {
      QuantumGatePtr gate = StandardQuantumGate::S(q);
      gate = cast::matmul(gate.get(),
                          StandardQuantumGate::S((q + 1) % NUM_QUBITS).get());
      gate = cast::matmul(gate.get(),
                          StandardQuantumGate::S((q + 2) % NUM_QUBITS).get());
      gates.push_back(gate);
    }
    break;
  }
  default:
    std::cerr << BOLDRED("[Error]: ")
              << "Unknown gate type: " << static_cast<int>(gateType) << "\n";
    std::exit(EXIT_FAILURE);
  }
} // createGates

// Number of threads to use for benchmarking.
int NUM_THREADS = cast::get_cpu_num_threads();
CPUSimdWidth SIMD_WIDTH = cast::get_cpu_simd_width();

static double computeGiBps(std::size_t memoryInBytes, double timeInSeconds) {
  return static_cast<double>(memoryInBytes) /
         (timeInSeconds * 1024 * 1024 * 1024);
}

template <typename ScalarType>
static void cpu_benchmark(const std::vector<GateType>& gateTypes,
                          const std::string& deviceName) {
  static_assert(std::is_same_v<ScalarType, float> ||
                    std::is_same_v<ScalarType, double>,
                "ScalarType must be either float or double");

  CPUKernelManager kernelMgr;
  CPUKernelGenConfig kernelGenConfig(
      SIMD_WIDTH,
      (std::is_same_v<ScalarType, float> ? Precision::F32 : Precision::F64));

  std::vector<QuantumGatePtr> gates;
  for (const auto& gateType : gateTypes)
    createGates(gates, gateType);

  for (int i = 0; i < gates.size(); ++i) {
    const auto& gate = gates[i];
    std::string funcName = "gate_" + std::to_string(i);
    kernelMgr.genStandaloneGate(kernelGenConfig, gate, funcName)
        .consumeError(); // ignore possible error
  }

  // Initialize JIT engine
  utils::timedExecute(
      [&]() {
        auto r = kernelMgr.initJIT(1, llvm::OptimizationLevel::O1, false, 1);
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
  timeit::Timer timer(/* replication */ 3, /* verbose */ 0);
  timeit::TimingResult tr;

  int gateIndex = 0;
  for (const auto& gateType : gateTypes) {
    std::vector<const CPUKernelInfo*> kernels;
    for (int q = 0; q < NUM_QUBITS; ++q) {
      std::string funcName = "gate_" + std::to_string(gateIndex++);
      const auto* kernelInfo = kernelMgr.getKernelByName(funcName);
      assert(kernelInfo != nullptr);
      kernels.push_back(kernelInfo);
    }

    tr = timer.timeit([&]() {
      for (auto* kernelInfo : kernels) {
        kernelMgr
            .applyCPUKernel(
                sv.data(), sv.nQubits(), kernelInfo->llvmFuncName, NUM_THREADS)
            .consumeError();
      }
    });
    // device,method,gate_type,nqubits,precision,time_per_gate
    std::cerr << deviceName << ",cast," << gateType << "," << NUM_QUBITS << ","
              << (std::is_same_v<ScalarType, float> ? "single" : "double")
              << "," << std::scientific << std::setprecision(6)
              << (tr.min / NUM_QUBITS) << "\n";
  }
}

#ifdef CAST_USE_CUDA
#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
template <typename ScalarType>
static void cuda_benchmark(const std::vector<GateType>& gateTypes,
                           const std::string& deviceName) {
  static_assert(std::is_same_v<ScalarType, float> ||
                    std::is_same_v<ScalarType, double>,
                "ScalarType must be either float or double");

  CUDAKernelManager kernelMgr;
  CUDAKernelGenConfig kernelGenConfig;

  std::vector<QuantumGatePtr> gates;
  for (const auto& gateType : gateTypes)
    createGates(gates, gateType);

  for (int i = 0; i < gates.size(); ++i) {
    const auto& gate = gates[i];
    std::string funcName = "gate_" + std::to_string(i);
    kernelMgr.genStandaloneGate(kernelGenConfig, gate, funcName)
        .consumeError(); // ignore possible error
  }

  // Initialize JIT engine
  utils::timedExecute(
      [&]() {
        auto r = kernelMgr.initJIT(1, llvm::OptimizationLevel::O1, false, 1);
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
  timeit::Timer timer(/* replication */ 3, /* verbose */ 0);
  timeit::TimingResult tr;

  int gateIndex = 0;
  for (const auto& gateType : gateTypes) {
    std::vector<const CPUKernelInfo*> kernels;
    for (int q = 0; q < NUM_QUBITS; ++q) {
      std::string funcName = "gate_" + std::to_string(gateIndex++);
      const auto* kernelInfo = kernelMgr.getKernelByName(funcName);
      assert(kernelInfo != nullptr);
      kernels.push_back(kernelInfo);
    }

    tr = timer.timeit([&]() {
      for (auto* kernelInfo : kernels) {
        kernelMgr
            .applyCPUKernel(
                sv.data(), sv.nQubits(), kernelInfo->llvmFuncName, NUM_THREADS)
            .consumeError();
      }
    });
    // device,method,gate_type,nqubits,precision,time_per_gate
    std::cerr << deviceName << ",cast," << gateType << "," << NUM_QUBITS << ","
              << (std::is_same_v<ScalarType, float> ? "single" : "double")
              << "," << std::scientific << std::setprecision(6)
              << (tr.min / NUM_QUBITS) << "\n";
  }
}

#endif // CAST_USE_CUDA

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << BOLDRED("[Error]: ") << "Usage: " << argv[0]
              << " <device_name>\n";
    return EXIT_FAILURE;
  }
  std::cerr << BOLDCYAN("[Info]: ")
            << "Benchmarking CPU simulation speed of multi-qubit dense gates "
            << "and multi-qubit Hadamard gates on CPU.\n";
  std::cerr << BOLDCYAN("[Info]: ") << "Using " << NUM_QUBITS
            << "-qubit statevector with " << NUM_THREADS << " threads.\n";
  std::cerr << BOLDCYAN("[Info]: ") << "SIMD width is set to "
            << static_cast<int>(SIMD_WIDTH) << " bits.\n";

  std::cerr << BOLDCYAN("[Info]: ") << "Starting single-precision test.\n";
  std::vector<GateType> gateTypes{U1, H1, S1, U3, H3, S3};

  std::string deviceName(argv[1]);
  cpu_benchmark<float>(gateTypes, deviceName);

  std::cerr << BOLDCYAN("[Info]: ") << "Starting double-precision test.\n";
  cpu_benchmark<double>(gateTypes, deviceName);
  return 0;
}