#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>

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


std::pair<double,double> measureJITOverhead(
  cast::CUDAKernelManager& kernelMgr,
  int nThreads,
  llvm::OptimizationLevel optLevel)
{
  using namespace timeit;

  Timer timer(3);
  TimingResult trPTX = timer.timeit([&]() {
    kernelMgr.emitPTX(nThreads, optLevel, /*verbose=*/0);
  });
  double ptxTimeSec = trPTX.med; // median time
  double ptxTimeMs  = ptxTimeSec * 1e3;

  TimingResult trJIT = timer.timeit([&]() {
    kernelMgr.initCUJIT(nThreads, /*verbose=*/0);
  });
  double jitTimeSec = trJIT.med;
  double jitTimeMs  = jitTimeSec * 1e3;

  return {ptxTimeMs, jitTimeMs};
}


int main() {
  using namespace cast;

  std::ofstream outFile("JITOverhead_bcmk.csv");
  if (!outFile.is_open()) {
    std::cerr << "Error: Could not open output file.\n";
    return 1;
  }

  std::vector<std::string> qasmFiles = {
    "../examples/qft/qft-16-cp.qasm",
    "../examples/qft/qft-28-cp.qasm"
  };

  std::vector<FusionStrategy> fusionStrategies = {
    FusionStrategy::NoFuse,
    FusionStrategy::Naive,
    FusionStrategy::Adaptive,
    FusionStrategy::Cuda
  };

  std::vector<int> naiveMaxKs = {2, 3};

  std::string adaptiveModelPath = "StandardModel.csv";
  std::string cudaModelPath     = "CUDAModel.csv";

  std::vector<int> blockSizes = {64, 128};
  std::vector<int> bitPrecisions = {32, 64};

  std::vector<CUDAKernelGenConfig::MatrixLoadMode> loadModes = {
    CUDAKernelGenConfig::UseMatImmValues,
    CUDAKernelGenConfig::LoadInDefaultMemSpace,
    CUDAKernelGenConfig::LoadInConstMemSpace
  };

  std::vector<llvm::OptimizationLevel> optLevels = {
    llvm::OptimizationLevel::O0,
    llvm::OptimizationLevel::O1,
    llvm::OptimizationLevel::O2
  };

  int nThreads = 4;

  outFile << "QASM_File,NumQubits,FusionStrategy,NaiveMaxK,BlockSize,"
             "Precision,LoadMode,OptLevel,PTXGenTime_ms,JITInitTime_ms\n";

  for (const auto& qasmFile : qasmFiles) {
    auto root = parseQasmFileOnce(qasmFile);
    if (!root) {
      std::cerr << "Failed to parse file " << qasmFile << "\n";
      continue;
    }

    // Loop strategies
    for (auto strategy : fusionStrategies) {
      for (int naiveMaxK : naiveMaxKs) {
        if (strategy != FusionStrategy::Naive && naiveMaxK != 2) {
          if (strategy != FusionStrategy::Naive) continue;
        }

        auto graphPtr = buildCircuitGraphInPlace(*root);
        if (!graphPtr) {
          std::cerr << "Failed to build circuit for " << qasmFile << "\n";
          continue;
        }
        cast::CircuitGraph& graph = *graphPtr;
        int nQubits = graph.nQubits;

        applyFusionStrategy(graph, strategy, naiveMaxK, adaptiveModelPath, cudaModelPath);

        for (auto blockSize : blockSizes) {
          for (auto bprec : bitPrecisions) {
            for (auto loadMode : loadModes) {
              for (auto optLevel : optLevels) {

                CUDAKernelManager kernelMgr;
                CUDAKernelGenConfig genConfig;
                genConfig.blockSize      = blockSize;
                genConfig.precision      = bprec;
                genConfig.matrixLoadMode = loadMode;

                kernelMgr.genCUDAGatesFromCircuitGraph(genConfig, graph, "testCircuit");

                auto [ptxMs, jitMs] = measureJITOverhead(kernelMgr, nThreads, optLevel);

                outFile << qasmFile << ","
                        << nQubits << ","
                        << toString(strategy) << ","
                        << naiveMaxK << ","
                        << blockSize << ","
                        << (bprec == 32 ? "Float32" : "Float64") << ",";

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

                outFile << ptxMs << "," << jitMs << "\n";

              } // optLevel
            }   // loadMode
          }     // bprec
        }       // blockSize
      }         // naiveMaxK
    }           // strategy
  }             // qasmFile

  outFile.close();
  return 0;
}
