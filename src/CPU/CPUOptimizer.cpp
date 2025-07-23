#include "cast/CPU/CPUOptimizer.h"
#include "cast/Core/ImplOptimize.h"

#include "utils/Formats.h"
#include "timeit/timeit.h"

using namespace cast;

void CPUOptimizer::run(ir::CircuitNode& circuit, utils::Logger logger) const {
  assert(fusionConfig != nullptr);
  // Step 1: Run canonicalization pass if enabled
  if (enableCanonicalization_) {
    // Canoicalization Block-wise Fusion Pass
    int nFused;
    double t = timeit::once([&]() {
      auto allCircuitGraphs = circuit.getAllCircuitGraphs();
      for (auto* graph : allCircuitGraphs)
        nFused = cast::impl::applySizeTwoFusion(*graph, fusionConfig->swapTol);
    });
    logger.log(1) << "Canoicalization Block-wise Fusion Pass: "
                  << nFused << " gates fused in "
                  << utils::fmt_time(t) << "\n";
    if (enableCFO_) {
        // Canonicalization CFO Fusion Pass
      double t = timeit::once([&]() {
        SizeOnlyFusionConfig cfoFusionConfig(2);
        cfoFusionConfig.swapTol = fusionConfig->swapTol;
        nFused = cast::impl::applyCFOFusion(circuit, &cfoFusionConfig, 2);
      });
      logger.log(1) << "Canonicalization CFO Fusion Pass: "
                    << nFused << " gates fused in "
                    << utils::fmt_time(t) << "\n";
    }
  }

  // Step 2: Run agglomerative fusion pass if enabled
  if (enableFusion_) {
    int nFused = 0;
    double t = timeit::once([&]() {
      auto allCircuitGraphs = circuit.getAllCircuitGraphs();
      for (int maxCandidateSize = fusionConfig->sizeMin;
          maxCandidateSize <= fusionConfig->sizeMax;
          ++maxCandidateSize) {
        int nFusedThisSize = 0;
        for (auto* graph : allCircuitGraphs) {
          int nFusedThisRound = 0;
          do {
            nFusedThisRound = impl::applyGateFusion(*graph,
                                                    fusionConfig.get(),
                                                    maxCandidateSize);
            nFusedThisSize += nFusedThisRound;
          } while (fusionConfig->enableMultiTraverse && nFusedThisRound > 0);
        }
        // Processed every graph in this size. Run CFO fusion if enabled
        if (enableCFO_ && nFusedThisSize > 0) {
          nFusedThisSize += impl::applyCFOFusion(circuit, 
                                                 fusionConfig.get(),
                                                 maxCandidateSize);
        }
        nFused += nFusedThisSize;
        logger.log(2) << " At size " << maxCandidateSize
                      << ", fused " << nFusedThisSize << " gates.\n";
      } // for maxCandidateSize
    });
    logger.log(1) << "Agglomerative Fusion Pass: Finished in "
                  << utils::fmt_time(t) << "\n";
  } // if enableFusion_
}

void CPUOptimizer::run(ir::CircuitGraphNode& graph,
                       utils::Logger logger) const {
  assert(fusionConfig != nullptr);
  // Step 1: Run canonicalization pass if enabled
  if (enableCanonicalization_) {
    int nFused = cast::impl::applySizeTwoFusion(graph, fusionConfig->swapTol);
    logger.log(1) << "Canoicalization Block-wise Fusion Pass: "
                  << nFused << " gates fused.\n";
  }

  // Step 2: Run agglomerative fusion pass if enabled
  if (enableFusion_) {
    int nFused = 0;
    double t = timeit::once([&]() {
      for (int maxCandidateSize = fusionConfig->sizeMin;
          maxCandidateSize <= fusionConfig->sizeMax;
          ++maxCandidateSize) {
        int nFusedThisRound = impl::applyGateFusion(graph,
                                                    fusionConfig.get(),
                                                    maxCandidateSize);
        nFused += nFusedThisRound;
        logger.log(2) << " At size " << maxCandidateSize
                      << ", fused " << nFusedThisRound << " gates.\n";
      } // for maxCandidateSize
    });
    logger.log(1) << "Agglomerative Fusion Pass: Finished in "
                  << utils::fmt_time(t) << "\n";
  } // if enableFusion_
}