#include "cast/CUDA/CUDAOptimizer.h"
#include "cast/Core/ImplOptimize.h"
#include "timeit/timeit.h"
#include "utils/Formats.h"

using namespace cast;

void CUDAOptimizer::run(ir::CircuitNode& circuit, utils::Logger logger) const {
  assert(fusionConfig && "CUDAOptimizer requires a CUDAFusionConfig");
  int nFused = 0;

  // 1) Canonicalization (size-2 fusion)
  if (enableCanonicalization_) {
    double t = timeit::once([&] {
      auto graphs = circuit.getAllCircuitGraphs();
      for (auto* g : graphs)
        nFused = cast::impl::applySizeTwoFusion(*g, fusionConfig->swapTol);
    });
    logger.log(1) << "Canonicalization Block-wise Fusion: " << nFused
                  << " gates fused in " << utils::fmt_time(t) << "\n";
  }

  // 2) Agglomerative fusion (+ optional CFO)
  if (enableFusion_) {
    double t = timeit::once([&] {
      for (int size = fusionConfig->sizeMin; size <= fusionConfig->sizeMax; ++size) {
        int fusedThisSize = 0;
        auto graphs = circuit.getAllCircuitGraphs();
        if (auto* cm = llvm::dyn_cast<CUDACostModel>(fusionConfig->costModel.get())) {
          cm->enableProbing(true);
          cm->resetProbeBudget(/*topK*/ 4);
          cm->setProbeThreads(std::max(1, /* pick a number */ 2));
          cm->setProbeOptLevel(llvm::OptimizationLevel::O1);

          auto p = cm->params;
          p.occAlpha = 0.5; // mild occupancy influence
          p.coalAlpha = 1.0; // stronger coalescing influence
          cm->setPenaltyParams(p);
        }
        for (auto* g : graphs) {
          int round = 0;
          do {
            round = cast::impl::applyGateFusion(*g, fusionConfig.get(), size);
            fusedThisSize += round;
          } while (fusionConfig->enableMultiTraverse && round > 0);
        }
        if (enableCFO_ && fusedThisSize > 0)
          fusedThisSize += cast::impl::applyCFOFusion(circuit, fusionConfig.get(), size);
        nFused += fusedThisSize;
        logger.log(2) << "  size " << size << ": +" << fusedThisSize << "\n";
      }
    });
    logger.log(1) << "Agglomerative Fusion finished in "
                  << utils::fmt_time(t) << "\n";
  }
}

void CUDAOptimizer::run(ir::CircuitGraphNode& graph, utils::Logger logger) const {
  assert(fusionConfig && "CUDAOptimizer requires a CUDAFusionConfig");

  // 1) Canonicalization (size-2 fusion)
  if (enableCanonicalization_) {
    int n = cast::impl::applySizeTwoFusion(graph, fusionConfig->swapTol);
    logger.log(1) << "Canonicalization Block-wise Fusion: " << n
                  << " gates fused.\n";
  }

  // 2) Agglomerative fusion
  if (enableFusion_) {
    double t = timeit::once([&] {
      for (int size = fusionConfig->sizeMin; size <= fusionConfig->sizeMax; ++size) {
        int n = cast::impl::applyGateFusion(graph, fusionConfig.get(), size);
        logger.log(2) << "  size " << size << ": +" << n << "\n";
      }
    });
    logger.log(1) << "Agglomerative Fusion finished in "
                  << utils::fmt_time(t) << "\n";
  }
}
