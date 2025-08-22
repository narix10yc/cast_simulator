#include "cast/CPU/CPUKernelManager.h"
#include "cast/CPU/CPUStatevector.h"
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
            .applyCPUKernel(sv.data(), sv.nQubits(), *kernelInfo, NUM_THREADS)
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
#include "cast/CUDA/CUDAPermute.h"
#include <algorithm>
#include <numeric>

static inline void set_layout_LSB(cast::BitLayout& layout,
                                  const std::vector<int>& Q,
                                  int nSys) {
  const int k = (int)Q.size();
  std::vector<char> isTarget(nSys, 0);
  for (int b = 0; b < k; ++b) isTarget[Q[b]] = 1;

  // non-targets in ascending old physical order
  std::vector<std::pair<int,int>> others;
  others.reserve(nSys - k);
  for (int l = 0; l < nSys; ++l)
    if (!isTarget[l]) others.emplace_back(layout.phys_of_log[l], l);
  std::sort(others.begin(), others.end());

  for (int b = 0; b < k; ++b) layout.phys_of_log[Q[b]] = b;
  for (int i = 0; i < (int)others.size(); ++i)
    layout.phys_of_log[others[i].second] = k + i;

  for (int p = 0; p < nSys; ++p) layout.log_of_phys[p] = -1;
  for (int l = 0; l < nSys; ++l) layout.log_of_phys[layout.phys_of_log[l]] = l;
}

template<typename ScalarType>
static inline void permute_to_LSBs_if_needed(
    ScalarType*& dCurrent,
    ScalarType*  dScratch,
    int          nSys,
    const std::vector<int>& logicalQubits,
    cast::BitLayout& layout,
    cudaStream_t stream = 0)
{
  const int k = (int)logicalQubits.size();
  if (k == 0) return;

  bool lsbOK = true;
  for (int b = 0; b < k; ++b)
    if (layout.phys_of_log[logicalQubits[b]] != b) { lsbOK = false; break; }
  if (lsbOK) return;

  uint64_t maskLow = 0;
  for (int b = 0; b < k; ++b)
    maskLow |= (1ull << layout.phys_of_log[logicalQubits[b]]);

  cast_permute_lowbits<ScalarType>(dCurrent, dScratch, nSys, maskLow, k, stream);
  std::swap(dCurrent, dScratch);

  set_layout_LSB(layout, logicalQubits, nSys);
}


template <typename ScalarType>
static void cuda_benchmark(const std::vector<GateType>& gateTypes,
                           const std::string& deviceName) {
  static_assert(std::is_same_v<ScalarType, float> ||
                    std::is_same_v<ScalarType, double>,
                "ScalarType must be either float or double");
  
  CUDAKernelManager kernelMgr;
  CUDAKernelGenConfig kernelGenConfig;
  kernelGenConfig.precision =
      (std::is_same_v<ScalarType, float> ? Precision::F32 : Precision::F64);

  kernelGenConfig.matrixLoadMode = CUDAMatrixLoadMode::UseMatImmValues;
  kernelGenConfig.assumeContiguousTargets = true;

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
        kernelMgr.emitPTX(1, llvm::OptimizationLevel::O1, 1);
        kernelMgr.initCUJIT(1, 1);
      },
      "Initialize JIT Engine");

  // Create a statevector with NUM_QUBITS qubits
  cast::CUDAStatevector<ScalarType> sv(NUM_QUBITS);
  utils::timedExecute([&]() { sv.randomize(); }, "Initialize statevector");

  BitLayout layout;
  layout.init(NUM_QUBITS);

  ScalarType* dCurrent = reinterpret_cast<ScalarType*>(sv.dData());
  ScalarType* dScratch = nullptr;
  {
    const uint64_t nComplex = 1ull << NUM_QUBITS;
    const size_t   nScalars = size_t(2) * nComplex;
    cudaMalloc(&dScratch, nScalars * sizeof(ScalarType));
  }

  timeit::Timer timer(/*replication*/3, /*verbose*/0);
  timeit::TimingResult tr;

  int gateIndex = 0;
  for (const auto& gateType : gateTypes) {
    std::vector<const CUDAKernelInfo*> kernels;
    for (int q = 0; q < NUM_QUBITS; ++q) {
      std::string funcName = "gate_" + std::to_string(gateIndex++);
      const auto* kernelInfo = kernelMgr.getKernelByName(funcName);
      assert(kernelInfo != nullptr);
      kernels.push_back(kernelInfo);
    }

    tr = timer.timeit([&]() {
      for (auto* kernelInfo : kernels) {
        const auto& Q = kernelInfo->gate->qubits();
        permute_to_LSBs_if_needed<ScalarType>(
            dCurrent, dScratch, sv.nQubits(), Q, layout, /*stream*/0);
        kernelMgr.launchCUDAKernel(static_cast<void*>(dCurrent),
                                   sv.nQubits(), *kernelInfo);
      }
      cudaDeviceSynchronize();
    });

    std::cerr << deviceName << ",cast," << gateType << "," << NUM_QUBITS << ","
              << (std::is_same_v<ScalarType, float> ? "single" : "double")
              << "," << std::scientific << std::setprecision(6)
              << (tr.min / NUM_QUBITS) << "\n";
  }
  cudaFree(dScratch);
}

#endif // CAST_USE_CUDA

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << BOLDRED("[Error]: ") << "Usage: " << argv[0]
              << "{CPU/GPU} <device_name>\n";
    return EXIT_FAILURE;
  }
  std::string cpu_or_gpu(argv[1]);
  if (cpu_or_gpu != "CPU" && cpu_or_gpu != "GPU") {
    std::cerr << BOLDRED("[Error]: ") << "Invalid argument: " << cpu_or_gpu
              << ". Expected 'CPU' or 'GPU'.\n";
    return EXIT_FAILURE;
  }
  if (cpu_or_gpu == "GPU") {
#ifdef CAST_USE_CUDA
    std::cerr << BOLDCYAN("[Info]: ") << "Starting CUDA benchmark.\n";
    std::cerr << "System information:\n";
    cast::displayCUDA();
    std::vector<GateType> gateTypes{U1, H1, S1, U3, H3, S3};
    std::cerr << BOLDCYAN("[Info]: ") << "Using " << NUM_QUBITS << "-qubit "
              << "statevectors\n";
    std::cerr << BOLDCYAN("[Info]: ") << "Starting single-precision test.\n";
    cuda_benchmark<float>(gateTypes, argv[2]);
    std::cerr << BOLDCYAN("[Info]: ") << "Starting double-precision test.\n";
    cuda_benchmark<double>(gateTypes, argv[2]);
    return 0;
#else
    std::cerr << BOLDRED("[Error]: ")
              << "CUDA support is not enabled in this build.\n";
    return EXIT_FAILURE;
#endif // CAST_USE_CUDA
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

  std::string deviceName(argv[2]);
  cpu_benchmark<float>(gateTypes, deviceName);

  std::cerr << BOLDCYAN("[Info]: ") << "Starting double-precision test.\n";
  cpu_benchmark<double>(gateTypes, deviceName);
  return 0;
}