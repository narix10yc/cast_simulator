#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <fstream>

#include "openqasm/parser.h"
#include "openqasm/ast.h"
#include "cast/CircuitGraph.h"
#include "cast/Fusion.h"
#include "simulation/KernelManager.h"
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

std::unique_ptr<openqasm::ast::RootNode> parseQasmFileOnce(const std::string& qasmFile) {
  openqasm::Parser parser(qasmFile, /*verbosity=*/0);

  auto root = parser.parse(); 
  return root;
}

std::unique_ptr<cast::CircuitGraph> buildCircuitGraphInPlace(
    const openqasm::ast::RootNode& root)
{
  auto graphPtr = std::make_unique<cast::CircuitGraph>();
  root.toCircuitGraph(*graphPtr);
  return graphPtr;
}

void applyFusionStrategy(
  cast::CircuitGraph& graph,
  FusionStrategy strategy,
  int naiveMaxK,
  const std::string& modelPath,
  const std::string& cudaModelPath)
{
  using namespace cast;

  FusionConfig fusionConfig = FusionConfig::Aggressive;
  fusionConfig.nThreads  = -1;
  fusionConfig.precision = 64;

  switch (strategy) {
    case FusionStrategy::NoFuse:
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


// Measure how long genCUDAGatesFromCircuitGraph(...) takes
double measureKernelGenerationTime(
  const cast::CircuitGraph& graph,
  int blockSize,
  PrecisionMode precision,
  cast::CUDAKernelGenConfig::MatrixLoadMode loadMode,
  llvm::OptimizationLevel optLevel)
{
  using namespace cast;

  CUDAKernelManager kernelMgr;
  CUDAKernelGenConfig genConfig;
  genConfig.blockSize      = blockSize;
  genConfig.precision      = (precision == PrecisionMode::Float32) ? 32 : 64;
  genConfig.matrixLoadMode = loadMode;

  timeit::Timer timer(/*replication=*/3);

  timeit::TimingResult tr = timer.timeit([&]() {
    kernelMgr.genCUDAGatesFromCircuitGraph(genConfig, graph, "testCircuit");
  });

  double timeSec = tr.med;
  double timeMs  = timeSec * 1.0e3;
  return timeMs;
}


int main() {
  using namespace cast;

  std::ofstream outFile("KernGen_bcmk.csv");
  if (!outFile.is_open()) {
    std::cerr << "Error: Could not open output file.\n";
    return 1;
  }

  std::vector<std::string> qasmFiles = {
    "../examples/qft/qft-16-cp.qasm",
    "../examples/qft/qft-28-cp.qasm",
    "../examples/qft/qft-35-cp.qasm"
  };

  std::vector<FusionStrategy> fusionStrategies = {
    FusionStrategy::NoFuse,
    FusionStrategy::Naive,
    FusionStrategy::Adaptive,
    FusionStrategy::Cuda
  };

  // For naive fusion
  std::vector<int> naiveMaxKs = {2, 3, 4, 5};

  std::string adaptiveModelPath = "cost_model_simd1.csv";
  std::string cudaModelPath     = "cost_model_cuda_128.csv";

  std::vector<int> blockSizes = {64, 128, 256};

  std::vector<PrecisionMode> precisions = {
    PrecisionMode::Float32,
    PrecisionMode::Float64
  };

  std::vector<CUDAKernelGenConfig::MatrixLoadMode> loadModes = {
    CUDAKernelGenConfig::LoadInDefaultMemSpace,
    CUDAKernelGenConfig::LoadInConstMemSpace,
    CUDAKernelGenConfig::UseMatImmValues
  };

  std::vector<llvm::OptimizationLevel> optLevels = {
    llvm::OptimizationLevel::O0,
    llvm::OptimizationLevel::O1,
    llvm::OptimizationLevel::O2,
    llvm::OptimizationLevel::O3
  };

  // CSV header
  outFile << "QASM_File,NumQubits,FusionStrategy,NaiveMaxK,BlockSize,"
               "Precision,LoadMode,OptLevel,KernelGenTime_ms\n";

  // Outer loop: QASM files
  for (const auto& qasmFile : qasmFiles) {

    // Parse QASM once -> unique_ptr<ast::RootNode>
    auto root = parseQasmFileOnce(qasmFile);
    if (!root) {
      std::cerr << "Failed to parse file " << qasmFile << "\n";
      continue;
    }

    // Loop over fusion strategies
    for (auto strategy : fusionStrategies) {
      // Loop over naiveMaxK
      for (int naiveMaxK : naiveMaxKs) {
        // If strategy != Naive, skip extra K
        if (strategy != FusionStrategy::Naive && naiveMaxK != 2) {
          if (strategy != FusionStrategy::Naive) continue;
        }

        // Build a fresh circuit from the AST
        auto graphPtr = buildCircuitGraphInPlace(*root);
        if (!graphPtr) {
          std::cerr << "Failed to build CircuitGraph\n";
          continue;
        }

        cast::CircuitGraph& graph = *graphPtr;
        int nQubits = graph.nQubits;

        // Apply chosen fusion
        applyFusionStrategy(graph, strategy, naiveMaxK, adaptiveModelPath, cudaModelPath);

        // Now iterate block sizes, precision, loadMode, optLevel
        for (auto blockSize : blockSizes) {
          for (auto prec : precisions) {
            for (auto loadMode : loadModes) {
              for (auto optLevel : optLevels) {

                double t_ms = measureKernelGenerationTime(
                  graph, blockSize, prec, loadMode, optLevel
                );

                outFile << qasmFile << ","
                          << nQubits << ","
                          << toString(strategy) << ","
                          << naiveMaxK << ","
                          << blockSize << ","
                          << toString(prec) << ",";

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

                // Print the measured time
                outFile << t_ms << "\n";
              }
            }
          }
        } // end blockSize loop
      }   // end naiveMaxK
    }     // end fusionStrategies
  }       // end qasmFiles

  return 0;
}
