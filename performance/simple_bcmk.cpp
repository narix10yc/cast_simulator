#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "openqasm/parser.h"
#include "openqasm/ast.h"
#include "cast/CircuitGraph.h"
#include "cast/Fusion.h"
#include "simulation/KernelManager.h"
#include "simulation/StatevectorCUDA.h"
#include "timeit/timeit.h"
#include "llvm/Passes/OptimizationLevel.h"

using namespace cast;

enum class FusionStrategy {
  NoFuse,
  Naive,
  Adaptive,
  Cuda
};

static const char* toString(FusionStrategy fs) {
  switch (fs) {
    case FusionStrategy::NoFuse:   return "NoFuse";
    case FusionStrategy::Naive:    return "Naive";
    case FusionStrategy::Adaptive: return "Adaptive";
    case FusionStrategy::Cuda:     return "Cuda";
    default:                       return "Unknown";
  }
}

static void printTimingStats(const std::string& label, const timeit::TimingResult& tr) {
  if (tr.tArr.empty()) {
    std::cout << label << ": No data\n";
    return;
  }
  double sum = 0.0;
  for (auto& t : tr.tArr) sum += t;
  double mean = sum / tr.tArr.size();
  double sumsq = 0.0;
  for (auto& t : tr.tArr) {
    double diff = t - mean;
    sumsq += diff * diff;
  }
  double var = (tr.tArr.size() > 1) ? (sumsq / tr.tArr.size()) : 0.0;
  double stdev = std::sqrt(var);

  auto computeMedian = [&](const std::vector<double>& arr) {
    if (arr.empty()) return 0.0;
    std::vector<double> sorted(arr);
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    if (n % 2 == 1) return sorted[n / 2];
    return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
  };
  double med = computeMedian(tr.tArr);

  double meanMs   = mean   * 1e3;
  double stdevMs  = stdev  * 1e3;
  double medianMs = med    * 1e3;

  std::cout << label << ": mean=" << meanMs << " ms ± " << stdevMs
            << " ms (median=" << medianMs << " ms, n=" << tr.tArr.size() << ")\n";
}


// Using identity matrix for now! (TODO)
void buildGateMatrixForBlock(unsigned fusedQubits, std::vector<double>& hostMatrix) {
  size_t dim = 1ULL << fusedQubits;
  // Fill everything with 0.0
  for (size_t i = 0; i < hostMatrix.size(); i++) {
    hostMatrix[i] = 0.0;
  }
  // Insert 1.0 on the diagonal
  for (size_t r = 0; r < dim; r++) {
    size_t base = 2ULL * (r * dim + r); // each matrix element has 2 doubles
    hostMatrix[base + 0] = 1.0; // real part of diagonal element
    hostMatrix[base + 1] = 0.0; // imaginary part is zero
  }
}

int main() {
  using namespace timeit;

  std::string qasmFile          = "../examples/qft/qft-16-cp.qasm";
  std::string adaptiveModelPath = "cost_model_simd1.csv";
  std::string cudaModelPath     = "cuda128test.csv";
  int naiveMaxK    = 2;
  int nReps        = 5;
  int blockSize    = 256;
  int nThreads     = 1;
  auto optLevel    = llvm::OptimizationLevel::O1;
  FusionStrategy strategy = FusionStrategy::Cuda;

  // Parse Time (ephemeral parse used for timing)
  Timer parseTimer(nReps);
  TimingResult parseTR = parseTimer.timeit([&]() {
    openqasm::Parser parser(qasmFile, 0);
    auto r = parser.parse();
  });

  // Parse again for the real AST we will use
  openqasm::Parser parser(qasmFile, 0);
  auto root = parser.parse();
  if (!root) {
    std::cerr << "Error: Could not parse.\n";
    return 1;
  }

  // Build Time (ephemeral CircuitGraph)
  Timer buildTimer(nReps);
  TimingResult buildTR = buildTimer.timeit([&]() {
    auto tmp = std::make_unique<cast::CircuitGraph>();
    root->toCircuitGraph(*tmp);
  });

  // Build the real CircuitGraph we will keep
  auto graphPtr = std::make_unique<cast::CircuitGraph>();
  root->toCircuitGraph(*graphPtr);

  // Prepare caches in unique_ptr so they remain alive
  std::unique_ptr<PerformanceCache> adaptiveCachePtr;
  std::unique_ptr<CUDAPerformanceCache> cudaCachePtr;
  if (!adaptiveModelPath.empty()) {
    // If LoadFromCSV() returns by value
    adaptiveCachePtr = std::make_unique<PerformanceCache>(
      PerformanceCache::LoadFromCSV(adaptiveModelPath)
    );
  }
  if (!cudaModelPath.empty()) {
    cudaCachePtr = std::make_unique<CUDAPerformanceCache>(
      CUDAPerformanceCache::LoadFromCSV(cudaModelPath)
    );
  }

  // Create cost-model objects **outside** ephemeral blocks so that
  // if the fusion or circuit references them, they do not go out of scope.
  NaiveCostModel     naiveModel(naiveMaxK, -1, 1e-8);
  StandardCostModel* scModelPtr = nullptr;
  CUDACostModel*     cudaModelPtr = nullptr;

  // create these on the heap if needed, so they live until main() returns.
  // An alternative is to store them as members in main scope (not pointers).
  std::unique_ptr<StandardCostModel> scModel;
  std::unique_ptr<CUDACostModel> cudaModel;
  if (adaptiveCachePtr) {
    scModel = std::make_unique<StandardCostModel>(adaptiveCachePtr.get());
    scModelPtr = scModel.get();
  }
  if (cudaCachePtr) {
    cudaModel = std::make_unique<CUDACostModel>(cudaCachePtr.get());
    cudaModelPtr = cudaModel.get();
  }

  // Fusion Time (ephemeral circuit used only for timing)
  Timer fusionTimer(nReps);
  TimingResult fusionTR = fusionTimer.timeit([&]() {
    auto tmp = std::make_unique<cast::CircuitGraph>();
    root->toCircuitGraph(*tmp);

    FusionConfig fusionConfig = FusionConfig::Aggressive;
    fusionConfig.nThreads  = -1;
    fusionConfig.precision = 64;

    switch (strategy) {
      case FusionStrategy::NoFuse:
        // No fusion
        break;

      case FusionStrategy::Naive:
        applyGateFusion(fusionConfig, &naiveModel, *tmp);
        break;

      case FusionStrategy::Adaptive:
        if (scModelPtr) {
          applyGateFusion(fusionConfig, scModelPtr, *tmp);
        }
        break;

      case FusionStrategy::Cuda:
        if (cudaModelPtr) {
          applyGateFusion(fusionConfig, cudaModelPtr, *tmp);
        }
        break;
    }
  });

  {
    FusionConfig fusionConfig = FusionConfig::Aggressive;
    fusionConfig.nThreads  = -1;
    fusionConfig.precision = 64;

    switch (strategy) {
      case FusionStrategy::NoFuse:
        break;

      case FusionStrategy::Naive:
        applyGateFusion(fusionConfig, &naiveModel, *graphPtr);
        break;

      case FusionStrategy::Adaptive:
        if (scModelPtr) {
          applyGateFusion(fusionConfig, scModelPtr, *graphPtr);
        }
        break;

      case FusionStrategy::Cuda:
        if (cudaModelPtr) {
          applyGateFusion(fusionConfig, cudaModelPtr, *graphPtr);
        }
        break;
    }
  }

  // Configure kernel generation
  CUDAKernelGenConfig genConfig;
  genConfig.blockSize      = blockSize;
  genConfig.precision      = 64;
  genConfig.matrixLoadMode = CUDAKernelGenConfig::LoadInDefaultMemSpace;

  // Kernel Gen Time (ephemeral test)
  Timer kernelGenTimer(nReps);
  TimingResult kernelGenTR = kernelGenTimer.timeit([&]() {
    CUDAKernelManager localKM;
    localKM.genCUDAGatesFromCircuitGraph(genConfig, *graphPtr, "testCircuit");
  });

  // PTX Time (ephemeral test)
  Timer ptxTimer(nReps);
  TimingResult ptxTR = ptxTimer.timeit([&]() {
    CUDAKernelManager localKM;
    localKM.genCUDAGatesFromCircuitGraph(genConfig, *graphPtr, "testCircuit");
    localKM.emitPTX(nThreads, optLevel, 0);
  });

  // JIT Time (ephemeral test)
  Timer jitTimer(nReps);
  TimingResult jitTR = jitTimer.timeit([&]() {
    CUDAKernelManager localKM;
    localKM.genCUDAGatesFromCircuitGraph(genConfig, *graphPtr, "testCircuit");
    localKM.emitPTX(nThreads, optLevel, 0);
    localKM.initCUJIT(nThreads, 0);
  });

  std::cout << "=== Post-Fusion Block Summary ===\n";
  for (const auto& block : graphPtr->getAllBlocks()) {
    auto gate = block->quantumGate;
    unsigned fusedQubits = gate->qubits.size();
    std::cout << "Block " << block->id 
              << ": covers " << fusedQubits << " qubits.\n";
    if (fusedQubits > 7) {
      std::cerr << "WARNING: Large fused gate with " 
                << fusedQubits << " qubits!\n";
    }
  }
  std::cout << "=================================\n";

  // Final kernel manager that we'll actually use for execution
  CUDAKernelManager kernelMgr;
  kernelMgr.genCUDAGatesFromCircuitGraph(genConfig, *graphPtr, "testCircuit");
  kernelMgr.emitPTX(nThreads, optLevel, 0);
  // llvm::outs() << "\n=== DUMPING PTX FOR INSPECTION ===\n";
  // kernelMgr.dumpPTX("bcmk_param_gate_test", llvm::outs());
  // llvm::outs() << "=== END PTX DUMP ===\n\n";
  
  kernelMgr.initCUJIT(nThreads, 0);

  auto kernels = kernelMgr.collectCUDAKernelsFromCircuitGraph("testCircuit");

  // Execution Time
  Timer execTimer(nReps);
  TimingResult execTR = execTimer.timeit([&]() {
    utils::StatevectorCUDA<double> sv(graphPtr->nQubits);
    sv.initialize();
    for (int i = 0; i < (int)kernels.size(); ++i) {
      auto* kInfo   = kernels[i];
      auto* block = graphPtr->getAllBlocks()[i];
      unsigned fusedQubits = block->quantumGate->qubits.size();
      size_t dim = (1ULL << fusedQubits);
      // auto gate = block->quantumGate;

      std::vector<double> hostMatrix(2ULL * dim * dim, 0.0);
      buildGateMatrixForBlock(fusedQubits, hostMatrix);

      double* dMatPtr = nullptr;
      size_t numElems = hostMatrix.size();
      cudaError_t err = cudaMalloc(&dMatPtr, numElems * sizeof(double));
      if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        continue; // or handle error
      }
      cudaMemcpy(dMatPtr, hostMatrix.data(),
                  hostMatrix.size() * sizeof(double),
                  cudaMemcpyHostToDevice);

      // kernelMgr.launchCUDAKernelParam(sv.dData(), sv.nQubits(), *k, blockSize);
      kernelMgr.launchCUDAKernelParam(
        sv.dData(),       // pointer to statevector
        sv.nQubits(),     // number of qubits
        *kInfo,           // the CUDAKernelInfo
        dMatPtr,          // pointer to matrix in GPU memory
        blockSize
      );
      cudaFree(dMatPtr);
    }
    cudaDeviceSynchronize();
  });

  // Print timing stats
  std::cout << "======== Timing Results ========\n";
  printTimingStats("1) Parse Time", parseTR);
  printTimingStats("2) Build Time", buildTR);
  printTimingStats("3) Fusion Time", fusionTR);
  printTimingStats("4) Kernel Gen Time", kernelGenTR);
  printTimingStats("5) PTX Time", ptxTR);
  printTimingStats("6) JIT Time", jitTR);
  printTimingStats("7) Execution Time", execTR);
  std::cout << "================================\n";

  return 0;
}
