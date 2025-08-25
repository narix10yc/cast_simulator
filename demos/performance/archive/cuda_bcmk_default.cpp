#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "cast/CUDA/CUDAStatevector.h"
#include "cast/Core/KernelManager.h"
#include "cast/Fusion.h"
#include "cast/Legacy/CircuitGraph.h"
#include "openqasm/AST.h"
#include "openqasm/Parser.h"
#include "timeit/timeit.h"
#include "llvm/Passes/OptimizationLevel.h"

using namespace cast;

enum class FusionStrategy { NoFuse, Naive, Adaptive, Cuda };

static const char* toString(FusionStrategy fs) {
  switch (fs) {
  case FusionStrategy::NoFuse:
    return "NoFuse";
  case FusionStrategy::Naive:
    return "Naive";
  case FusionStrategy::Adaptive:
    return "Adaptive";
  case FusionStrategy::Cuda:
    return "Cuda";
  default:
    return "Unknown";
  }
}

static void printTimingStats(const std::string& label,
                             const timeit::TimingResult& tr) {
  if (tr.tArr.empty()) {
    std::cout << label << ": No data\n";
    return;
  }
  double sum = 0.0;
  for (auto& t : tr.tArr)
    sum += t;
  double mean = sum / tr.tArr.size();
  double sumsq = 0.0;
  for (auto& t : tr.tArr) {
    double diff = t - mean;
    sumsq += diff * diff;
  }
  double var = (tr.tArr.size() > 1) ? (sumsq / tr.tArr.size()) : 0.0;
  double stdev = std::sqrt(var);

  auto computeMedian = [&](const std::vector<double>& arr) {
    if (arr.empty())
      return 0.0;
    std::vector<double> sorted(arr);
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    if (n % 2 == 1)
      return sorted[n / 2];
    return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
  };
  double med = computeMedian(tr.tArr);

  double meanMs = mean * 1e3;
  double stdevMs = stdev * 1e3;
  double medianMs = med * 1e3;

  std::cout << label << ": mean=" << meanMs << " ms ± " << stdevMs
            << " ms (median=" << medianMs << " ms, n=" << tr.tArr.size()
            << ")\n";
}

int main() {
  using namespace timeit;

  // std::string qasmFile = "../examples/rqc/q12_189_128.qasm";
  // std::string qasmFile = "../examples/rqc/q20_592_427.qasm";
  std::string qasmFile = "../examples/rqc/q30_521_379.qasm";
  std::string adaptiveModelPath = "cost_model_simd1.csv";
  std::string cudaModelPath = "cuda128test.csv";
  int naiveMaxK = 2;
  int nReps = 2;
  int blockSize = 256;
  int nThreads = 1;
  auto optLevel = llvm::OptimizationLevel::O1;
  FusionStrategy strategy = FusionStrategy::Cuda;

  // Parse Time (ephemeral parse used for timing)
  Timer parseTimer(nReps);
  TimingResult parseTR = parseTimer.timeit([&]() {
    openqasm::Parser parser(qasmFile, 0);
    auto r = parser.parse();
  });

  // printTimingStats("1) Parse Time", parseTR);

  // Parse again for the real AST that will be used
  openqasm::Parser parser(qasmFile, 0);
  auto root = parser.parse();
  if (!root) {
    std::cerr << "Error: Could not parse.\n";
    return 1;
  }

  // Build Time (ephemeral legacy::CircuitGraph)
  Timer buildTimer(nReps);
  TimingResult buildTR = buildTimer.timeit([&]() {
    auto tmp = std::make_unique<cast::legacy::CircuitGraph>();
    root->tolegacy::CircuitGraph(*tmp);
  });

  // printTimingStats("2) Build Time", buildTR);

  // Build the real legacy::CircuitGraph we will keep
  auto graphPtr = std::make_unique<cast::legacy::CircuitGraph>();
  root->tolegacy::CircuitGraph(*graphPtr);

  // Prepare caches in unique_ptr so they remain alive
  std::unique_ptr<PerformanceCache> adaptiveCachePtr;
  std::unique_ptr<CUDAPerformanceCache> cudaCachePtr;
  if (!adaptiveModelPath.empty()) {
    // If LoadFromCSV() returns by value
    adaptiveCachePtr = std::make_unique<PerformanceCache>(
        PerformanceCache::LoadFromCSV(adaptiveModelPath));
  }
  if (!cudaModelPath.empty()) {
    cudaCachePtr = std::make_unique<CUDAPerformanceCache>(
        CUDAPerformanceCache::LoadFromCSV(cudaModelPath));
  }

  // Create cost-model objects **outside** ephemeral blocks so that
  // if the fusion or circuit references them, they do not go out of scope.
  NaiveCostModel naiveModel(naiveMaxK, -1, 1e-8);
  StandardCostModel* scModelPtr = nullptr;
  CUDACostModel* cudaModelPtr = nullptr;

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
    auto tmp = std::make_unique<cast::legacy::CircuitGraph>();
    root->tolegacy::CircuitGraph(*tmp);

    FusionConfig fusionConfig = FusionConfig::Aggressive;
    fusionConfig.nThreads = -1;
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

  // printTimingStats("3) Fusion Time", fusionTR);

  {
    FusionConfig fusionConfig = FusionConfig::Aggressive;
    fusionConfig.nThreads = -1;
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
  genConfig.blockSize = blockSize;
  genConfig.precision = 64;
  genConfig.matrixLoadMode = CUDAKernelGenConfig::LoadInDefaultMemSpace;

  // Kernel Gen Time (ephemeral test)
  Timer kernelGenTimer(nReps);
  TimingResult kernelGenTR = kernelGenTimer.timeit([&]() {
    CUDAKernelManager localKM;
    localKM.genCUDAGatesFromlegacy::CircuitGraph(
        genConfig, *graphPtr, "testCircuit");
  });

  // printTimingStats("4) Kernel Gen Time", kernelGenTR);

  // PTX Time (ephemeral test)
  Timer ptxTimer(nReps);
  std::unique_ptr<CUDAKernelManager> localKMptx;

  TimingResult ptxTR = ptxTimer.timeitPartial(
      // preMethod:
      [&]() {
        localKMptx = std::make_unique<CUDAKernelManager>();
        localKMptx->genCUDAGatesFromlegacy::CircuitGraph(
            genConfig, *graphPtr, "testCircuit");
      },
      // timedMethod: measure initCUJIT
      [&]() { localKMptx->emitPTX(nThreads, optLevel, 0); },
      // postMethod:
      [&]() { localKMptx.reset(); },
      // setup:
      []() {},
      // teardown:
      []() {});

  // printTimingStats("5) PTX Time", ptxTR);

  Timer jitTimer(nReps);
  std::unique_ptr<CUDAKernelManager> localKM;

  TimingResult jitTR = jitTimer.timeitPartial(
      // preMethod:
      [&]() {
        localKM = std::make_unique<CUDAKernelManager>();
        localKM->genCUDAGatesFromlegacy::CircuitGraph(
            genConfig, *graphPtr, "testCircuit");
        localKM->emitPTX(nThreads, optLevel, 0);
      },
      // timedMethod: measure initCUJIT
      [&]() { localKM->initCUJIT(nThreads, 0); },

      // postMethod:
      [&]() { localKM.reset(); },

      // setup:
      []() {},

      // teardown:
      []() {});

  // printTimingStats("6) JIT Time", jitTR);

  std::cout << "=== Post-Fusion Block Summary ===\n";
  for (const auto& block : graphPtr->getAllBlocks()) {
    auto gate = block->quantumGate;
    unsigned fusedQubits = gate->qubits.size();
    std::cout << "Block " << block->id << ": covers " << fusedQubits
              << " qubits.\n";
    if (fusedQubits > 7) {
      std::cerr << "WARNING: Large fused gate with " << fusedQubits
                << " qubits!\n";
    }
  }
  std::cout << "=================================\n";

  // Final kernel manager that we'll actually use for execution
  CUDAKernelManager kernelMgr;
  kernelMgr.genCUDAGatesFromlegacy::CircuitGraph(
      genConfig, *graphPtr, "testCircuit");
  kernelMgr.emitPTX(nThreads, optLevel, 0);

  kernelMgr.initCUJIT(nThreads, 0);
  // for (auto &kInfo : kernelMgr.kernels()) {
  //   std::string fnName = kInfo.llvmFuncName;
  //   llvm::outs() << "\n=== PTX for " << fnName << " ===\n";
  //   kernelMgr.dumpPTX(fnName, llvm::outs());
  // }

  auto kernels =
      kernelMgr.collectCUDAKernelsFromlegacy::CircuitGraph("testCircuit");

  // Execution Time
  Timer execTimer(nReps);
  TimingResult execTR = execTimer.timeit([&]() {
    cast::CUDAStatevector<double> sv(graphPtr->nQubits);
    sv.initialize();
    for (int i = 0; i < (int)kernels.size(); ++i) {
      auto* kInfo = kernels[i];
      auto* block = graphPtr->getAllBlocks()[i];
      unsigned fusedQubits = block->quantumGate->qubits.size();
      size_t dim = (1ULL << fusedQubits);
      // auto gate = block->quantumGate;

      std::vector<double> hostMatrix(2ULL * dim * dim, 0.0);
      // buildlegacy::GateMatrixForBlock(fusedQubits, hostMatrix);
      auto gateMatPtr = block->quantumGate->gateMatrix.getConstantMatrix();
      if (!gateMatPtr) {
        // with parametric gates, this will be more complex
        std::cerr << "Gate matrix not constant or not supported.\n";
        continue;
      }
      auto* cmplxData = gateMatPtr->data();

      size_t totalElems = (size_t)dim * (size_t)dim; // K*K
      // std::cerr << "Gate block " << block->id << ": fusedQubits=" <<
      // fusedQubits << "\n"; std::cerr << "  Some matrix entries:\n"; for (int
      // debugIdx = 0; debugIdx < std::min<size_t>(5, totalElems); debugIdx++) {
      //     std::cerr << "    [" << debugIdx << "]: "
      //               << cmplxData[debugIdx].real() << " + i"
      //               << cmplxData[debugIdx].imag() << "\n";
      // }
      for (size_t j = 0; j < totalElems; j++) {
        hostMatrix[2 * j + 0] = cmplxData[j].real();
        hostMatrix[2 * j + 1] = cmplxData[j].imag();
      }

      double* dMatPtr = nullptr;
      size_t numElems = hostMatrix.size();
      cudaError_t err = cudaMalloc(&dMatPtr, numElems * sizeof(double));
      if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        continue; // or handle error
      }
      cudaMemcpy(dMatPtr,
                 hostMatrix.data(),
                 hostMatrix.size() * sizeof(double),
                 cudaMemcpyHostToDevice);

      kernelMgr.launchCUDAKernelParam(
          sv.dData(),   // pointer to statevector
          sv.nQubits(), // number of qubits
          *kInfo,       // the CUDAKernelInfo
          dMatPtr,      // pointer to matrix in GPU memory
          blockSize);
      cudaFree(dMatPtr);
    }
    cudaDeviceSynchronize();
  });

  // printTimingStats("7) Execution Time", execTR);

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
