#include "cast/Core/Optimize.h"
#include "cast/Core/ImplOptimize.h"
#include "utils/iocolor.h"
#include "utils/PrintSpan.h"

#include "llvm/Support/Casting.h"

#define DEBUG_TYPE "fusion-new"
#include "llvm/Support/Debug.h"
// #define LLVM_DEBUG(X) X

using namespace cast;

void cast::applyGateFusionPass(ir::CircuitNode& circuit,
                               const FusionConfig* config) {
  auto allCircuitGraphs = circuit.getAllCircuitGraphs();
  for (int maxCandidateSize = config->sizeMin;
       maxCandidateSize <= config->sizeMax;
       ++maxCandidateSize) {
    int nFusedThisSize = 0;
    for (auto* graph : allCircuitGraphs) {
      int nFusedThisRound = 0;
      do {
        nFusedThisRound =
          impl::applyGateFusion(*graph, config, maxCandidateSize);
        nFusedThisSize += nFusedThisRound;
      } while (config->enableMultiTraverse && nFusedThisRound > 0);
    }
    // Processed every graph in this size. Run CFO fusion if enabled
    if (config->enableFusionCFOPass && nFusedThisSize > 0)
      nFusedThisSize += impl::applyCFOFusion(circuit, config, maxCandidateSize);
  }
}

void cast::applyCanonicalizationPass(ir::CircuitNode& circuit, double swapTol) {
  // perform fusion in each block
  auto allCircuitGraphs = circuit.getAllCircuitGraphs();
  for (auto* graph : allCircuitGraphs)
    cast::impl::applySizeTwoFusion(*graph, swapTol);

  // circuit.displayInfo(std::cerr << "\nAfter Size-2 Fusion\n", 1);
  // circuit.visualize(std::cerr) << "\n";

  SizeOnlyFusionConfig cfoFusionConfig(2);
  cfoFusionConfig.swapTol = swapTol;
  cast::impl::applyCFOFusion(circuit, &cfoFusionConfig, 2);
}