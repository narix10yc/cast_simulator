#include "cast/Core/Optimizer.h"
#include "cast/Detail/ImplOptimize.h"

#include "timeit/timeit.h"
#include "utils/Formats.h"

using namespace cast;

void OptimizerBase::run(ir::CircuitNode& circuit, utils::Logger logger) const {
  assert(fusionConfig_ != nullptr);

  // Step 1: Run canonicalization pass if enabled
  if (enableCanonicalization_) {
    // Canoicalization Block-wise Fusion Pass
    int nFused;
    double t = timeit::once([&]() {
      auto allCircuitGraphs = circuit.getAllCircuitGraphs();
      for (auto* graph : allCircuitGraphs)
        nFused = cast::impl::applySizeTwoFusion(*graph, fusionConfig_->swapTol);
    });
    logger.log(1) << "Canoicalization Block-wise Fusion Pass: " << nFused
                  << " gates fused in " << utils::fmt_time(t) << "\n";
    if (enableCFO_) {
      // Canonicalization CFO Fusion Pass
      double t = timeit::once([&]() {
        SizeOnlyFusionConfig cfoFusionConfig(2);
        cfoFusionConfig.swapTol = fusionConfig_->swapTol;
        nFused = cast::impl::applyCFOFusion(circuit, &cfoFusionConfig, 2);
      });
      logger.log(1) << "Canonicalization CFO Fusion Pass: " << nFused
                    << " gates fused in " << utils::fmt_time(t) << "\n";
    }
  }

  // Step 2: Run agglomerative fusion pass if enabled
  if (enableFusion_) {
    int nFused = 0;
    double t = timeit::once([&]() {
      auto allCircuitGraphs = circuit.getAllCircuitGraphs();
      for (int cddSize = fusionConfig_->sizeMin;
           cddSize <= fusionConfig_->sizeMax;
           ++cddSize) {
        int nFusedThisSize = 0;
        for (auto* graph : allCircuitGraphs) {
          int nFusedThisRound = 0;
          do {
            nFusedThisRound =
                impl::applyGateFusion(*graph, fusionConfig_.get(), cddSize);
            nFusedThisSize += nFusedThisRound;
          } while (fusionConfig_->enableMultiTraverse && nFusedThisRound > 0);
        } // for each graph
        // Processed every graph in this size. Run CFO fusion if enabled
        if (enableCFO_ && nFusedThisSize > 0) {
          nFusedThisSize +=
              impl::applyCFOFusion(circuit, fusionConfig_.get(), cddSize);
        }
        nFused += nFusedThisSize;
        logger.log(2) << " At size " << cddSize << ", fused " << nFusedThisSize
                      << " gates.\n";
      } // for candidate size (cddSize)
    });
    logger.log(1) << "Agglomerative Fusion Pass: Finished in "
                  << utils::fmt_time(t) << "\n";
  } // if enableFusion_
}

void OptimizerBase::run(ir::CircuitGraphNode& graph,
                        utils::Logger logger) const {
  assert(fusionConfig_ != nullptr);
  // Step 1: Run canonicalization pass if enabled
  if (enableCanonicalization_) {
    int nFused = 0;
    auto t = timeit::once([&]() {
      nFused = cast::impl::applySizeTwoFusion(graph, fusionConfig_->swapTol);
    });
    logger.log(1) << "(" << utils::fmt_time(t)
                  << ") Canonicalization Block-wise Fusion Pass: " << nFused
                  << " gates fused.\n";
  }

  // Step 2: Run agglomerative fusion pass if enabled
  if (enableFusion_) {
    int sMin = fusionConfig_->sizeMin;
    if (sMin == 2 && enableCanonicalization_)
      sMin = 3; // skip size 2 if canonicalization already done
    int sMax = fusionConfig_->sizeMax;

    for (int cddSize = sMin; cddSize <= sMax; ++cddSize) {
      int nFused;
      auto t = timeit::once([&]() {
        nFused = impl::applyGateFusion(graph, fusionConfig_.get(), cddSize);
      });
      logger.log(2) << "(" << utils::fmt_time(t) << ") At size " << cddSize
                    << ", fused " << nFused << " gates.\n";
    } // for maxCandidateSize

    logger.log(1) << "Agglomerative Fusion Pass finished: " << graph.nGates()
                  << " remains.\n";
  } // if enableFusion_
}
