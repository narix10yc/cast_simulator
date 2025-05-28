#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <fstream>

#include "openqasm/parser.h"
#include "openqasm/ast.h"
#include "cast/Legacy/CircuitGraph.h"
#include "cast/Fusion.h"
#include "cast/Core/KernelManager.h"
#include "cast/CUDA/StatevectorCUDA.h"
#include "timeit/timeit.h"
#include "llvm/Passes/OptimizationLevel.h"


enum class FusionStrategy {
  NoFuse,
  Naive,
  Adaptive,
  Cuda
};

enum class PrecisionMode {
  Float32,
  Float64
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

static const char* toString(PrecisionMode pm) {
  switch (pm) {
    case PrecisionMode::Float32: return "Float32";
    case PrecisionMode::Float64: return "Float64";
    default:                     return "Unknown";
  }
}


// Parse QASM => returns a unique_ptr<openqasm::ast::RootNode>
std::unique_ptr<openqasm::ast::RootNode> parseQasmFileOnce(const std::string& qasmFile) {
  openqasm::Parser parser(qasmFile, /*verbosity=*/0);
  auto root = parser.parse(); 
  return root;
}

// Build a legacy::CircuitGraph in place (stored in unique_ptr to avoid copies)
std::unique_ptr<cast::legacy::CircuitGraph> buildlegacy::CircuitGraphInPlace(
    const openqasm::ast::RootNode& root)
{
  auto graphPtr = std::make_unique<cast::legacy::CircuitGraph>();
  root.tolegacy::CircuitGraph(*graphPtr);
  return graphPtr;
}


// Apply the chosen fusion strategy
void applyFusionStrategy(
  cast::legacy::CircuitGraph& graph,
  FusionStrategy strategy,
  int naiveMaxK,
  const std::string& modelPath,
  const std::string& cudaModelPath)
{
  using namespace cast;

  FusionConfig fusionConfig = FusionConfig::Aggressive;
  fusionConfig.nThreads  = -1;  // e.g., GPU
  fusionConfig.precision = 64;  // or 32 for float

  switch (strategy) {
    case FusionStrategy::NoFuse:
      // do nothing
      break;

    case FusionStrategy::Naive: {
      NaiveCostModel naiveModel(naiveMaxK, -1, 1e-8);
      applyGateFusion(fusionConfig, &naiveModel, graph);
      break;
    }

    case FusionStrategy::Adaptive: {
      if (!modelPath.empty()) {
        auto cache = PerformanceCache::LoadFromCSV(modelPath);
        StandardCostModel scModel(&cache);
        applyGateFusion(fusionConfig, &scModel, graph);
      } else {
        std::cerr << "[Warning] Adaptive requested but no modelPath provided.\n";
      }
      break;
    }

    case FusionStrategy::Cuda: {
      if (!cudaModelPath.empty()) {
        auto cudaCache = CUDAPerformanceCache::LoadFromCSV(cudaModelPath);
        CUDACostModel cudaModel(&cudaCache);
        applyGateFusion(fusionConfig, &cudaModel, graph);
      } else {
        std::cerr << "[Warning] CUDA requested but no cudaModelPath provided.\n";
      }
      break;
    }
  }
}

// Measure how long it takes to execute the circuit.
double measureKernelExecutionTime(
  const cast::legacy::CircuitGraph& graph,
  int blockSize,
  PrecisionMode precision,
  cast::CUDAKernelGenConfig::MatrixLoadMode loadMode,
  llvm::OptimizationLevel optLevel)
{
  using namespace cast;

  // 1) Create a KernelManager + config
  CUDAKernelManager kernelMgr;
  CUDAKernelGenConfig genConfig;
  genConfig.blockSize      = blockSize;
  genConfig.precision      = (precision == PrecisionMode::Float32) ? 32 : 64;
  genConfig.matrixLoadMode = loadMode;

  kernelMgr.genCUDAGatesFromlegacy::CircuitGraph(genConfig, graph, "testCircuit");

  kernelMgr.emitPTX(/*nThreads=*/1, optLevel, /*verbose=*/0);
  kernelMgr.initCUJIT(/*nThreads=*/1, /*verbose=*/0);

  auto kernels = kernelMgr.collectCUDAKernelsFromlegacy::CircuitGraph("testCircuit");

  int nQubits = graph.nQubits;
  utils::StatevectorCUDA<double> sv(nQubits);
  sv.initialize(); // or randomize, etc.

  timeit::Timer timer(/*replications=*/3);
  timeit::TimingResult tr = timer.timeit([&]() {
    for (auto* kernel : kernels) {
      kernelMgr.launchCUDAKernel(
        sv.dData(),
        sv.nQubits(),
        *kernel,
        blockSize
      );
    }
    cudaDeviceSynchronize();
  });

  double timeSec = tr.med;
  double timeMs  = timeSec * 1.0e3;
  return timeMs;
}


int main() {
  using namespace cast;

  std::ofstream outFile("KernExec_bcmk.csv");
  if (!outFile.is_open()) {
    std::cerr << "Error: Could not open output file.\n";
    return 1;
  }

  // Example QASM input circuits
  std::vector<std::string> qasmFiles = {
    "../examples/qft/qft-16-cp.qasm",
    "../examples/qft/qft-28-cp.qasm",
    "../examples/qft/qft-35-cp.qasm"
  };

  // Fusion strategies
  std::vector<FusionStrategy> fusionStrategies = {
    FusionStrategy::NoFuse,
    FusionStrategy::Naive,
    FusionStrategy::Adaptive,
    FusionStrategy::Cuda
  };

  // For naive fusion
  std::vector<int> naiveMaxKs = {2, 3, 4, 5};

  std::string adaptiveModelPath = "cost_model_simd1.csv";
  std::string cudaModelPath = "cost_model_cuda_128.csv";

  // Various block sizes
  std::vector<int> blockSizes = {64, 128, 256};

  // Single vs. double precision
  std::vector<PrecisionMode> precisions = {
    PrecisionMode::Float32,
    PrecisionMode::Float64
  };

  // Matrix load modes
  std::vector<CUDAKernelGenConfig::MatrixLoadMode> loadModes = {
    CUDAKernelGenConfig::LoadInDefaultMemSpace,
    CUDAKernelGenConfig::LoadInConstMemSpace,
    CUDAKernelGenConfig::UseMatImmValues
  };

  // Different LLVM opt levels
  std::vector<llvm::OptimizationLevel> optLevels = {
    llvm::OptimizationLevel::O0,
    llvm::OptimizationLevel::O1,
    llvm::OptimizationLevel::O2,
    llvm::OptimizationLevel::O3
  };

  outFile << "QASM_File,NumQubits,FusionStrategy,NaiveMaxK,BlockSize,"
             "Precision,LoadMode,OptLevel,KernelExecTime_ms\n";

  // Outer loop over QASM files
  for (const auto& qasmFile : qasmFiles) {

    // Parse QASM once
    auto root = parseQasmFileOnce(qasmFile);
    if (!root) {
      std::cerr << "Failed to parse file " << qasmFile << "\n";
      continue;
    }

    // For each fusion strategy
    for (auto strategy : fusionStrategies) {

      // For each naiveMaxK
      for (int naiveMaxK : naiveMaxKs) {
        // If strategy != Naive, skip big K
        if (strategy != FusionStrategy::Naive && naiveMaxK != 2) {
          if (strategy != FusionStrategy::Naive) continue;
        }

        // Build a fresh circuit
        auto graphPtr = buildlegacy::CircuitGraphInPlace(*root);
        if (!graphPtr) {
          std::cerr << "Failed to build legacy::CircuitGraph\n";
          continue;
        }

        cast::legacy::CircuitGraph& graph = *graphPtr;
        int nQubits = graph.nQubits;

        // Apply fusion
        applyFusionStrategy(graph, strategy, naiveMaxK,
                            adaptiveModelPath, cudaModelPath);

        // Now loop over blockSizes, precision, loadMode, optLevel
        for (auto blockSize : blockSizes) {
          for (auto prec : precisions) {
            for (auto loadMode : loadModes) {
              for (auto optLevel : optLevels) {

                // Time the execution
                double t_ms = measureKernelExecutionTime(
                  graph, blockSize, prec, loadMode, optLevel
                );

                // Output a CSV row
                outFile << qasmFile << ","
                        << nQubits << ","
                        << toString(strategy) << ","
                        << naiveMaxK << ","
                        << blockSize << ","
                        << toString(prec) << ",";

                // Convert loadMode to string
                switch (loadMode) {
                  case CUDAKernelGenConfig::LoadInDefaultMemSpace:
                    outFile << "DefaultMem,";
                    break;
                  case CUDAKernelGenConfig::LoadInConstMemSpace:
                    outFile << "ConstMem,";
                    break;
                  case CUDAKernelGenConfig::UseMatImmValues:
                    outFile << "ImmValues,";
                    break;
                  default:
                    outFile << "UnknownLoadMode,";
                    break;
                }

                // Convert optLevel to string via if/else
                if (optLevel == llvm::OptimizationLevel::O0) {
                  outFile << "O0,";
                } else if (optLevel == llvm::OptimizationLevel::O1) {
                  outFile << "O1,";
                } else if (optLevel == llvm::OptimizationLevel::O2) {
                  outFile << "O2,";
                } else if (optLevel == llvm::OptimizationLevel::O3) {
                  outFile << "O3,";
                } else {
                  outFile << "UnknownOpt,";
                }

                outFile << t_ms << "\n";
              }
            }
          }
        } // end blockSizes
      } // end naiveMaxK
    } // end fusionStrategies
  } // end qasmFiles

  outFile.close();
  return 0;
}
