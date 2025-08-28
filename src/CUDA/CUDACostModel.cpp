#include "cast/CUDA/CUDACostModel.h"
#include "cast/Core/Config.h"

#include "utils/utils.h"

using namespace cast;

using WeightType = std::array<float, GLOBAL_MAX_GATE_SIZE>;

static void runPreliminaryExperiments(const CUDAKernelGenConfig& kernelConfig,
                                      int nQubits,
                                      int nWorkerThreads,
                                      int nRuns,
                                      int verbose,
                                      WeightType& weights) {
  CUDAKernelManager km(nWorkerThreads);

  struct GateNamePair {
    QuantumGatePtr gate;
    std::string name;
  };

  std::vector<GateNamePair> gateNamePairs;
  const auto generateGatesAndInitJit = [&]() {
    gateNamePairs.reserve(5);
    // generate {1,2,3,4,5}-qubit gates acting on MSB qubits
    for (int k = 1; k <= 5; ++k) {
      QuantumGate::TargetQubitsType qubits;
      qubits.reserve(k);
      for (int i = 0; i < k; ++i)
        qubits.push_back(nQubits - i);
      gateNamePairs.emplace_back(StandardQuantumGate::RandomUnitary(qubits),
                                 "gate_k" + std::to_string(k));
      km.genStandaloneGate(
            kernelConfig, gateNamePairs.back().gate, gateNamePairs.back().name)
          .consumeError();
    }
    // only show progress bar if verbose >= 2
    km.compileLLVMIRToPTX(1, verbose - 1);
    km.compilePTXToCubin(1, verbose - 1);
  };

  if (verbose >= 1) {
    utils::timedExecute(generateGatesAndInitJit,
                        "Code Generation and JIT Initialization");
  } else {
    generateGatesAndInitJit();
  }
}

void CUDAPerformanceCache::runExperiments(
    const CUDAKernelGenConfig& kernelConfig,
    int nQubits,
    int nWorkerThreads,
    int nRuns,
    int verbose) {
  WeightType weights;
  runPreliminaryExperiments(
      kernelConfig, nQubits, nWorkerThreads, nRuns, verbose, weights);
}