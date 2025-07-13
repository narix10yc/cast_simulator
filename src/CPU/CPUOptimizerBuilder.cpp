#include "cast/CPU/CPUOptimizerBuilder.h"
#include "cast/Core/ImplOptimize.h"

using namespace cast;

MaybeError<Optimizer> CPUOptimizerBuilder::build() {
  Optimizer opt;
  if (data->enableCanonicalization) {
    opt.addPass(
      "Canoicalization Block-wise Fusion Pass",
      [swapTol=data->fusionConfig->swapTol,
       verbose=data->verbose](ir::CircuitNode& circuit) { 
        auto allCircuitGraphs = circuit.getAllCircuitGraphs();
        int nFused;
        for (auto* graph : allCircuitGraphs)
          nFused = cast::impl::applySizeTwoFusion(*graph, swapTol);
        if (verbose > 0) {
          std::cerr << "Size-2 fusion pass: " << nFused << " gates fused.\n";
        }
    });
    if (data->enableCFO) {
      opt.addPass(
        "Canonicalization CFO Fusion Pass",
        [swapTol=data->fusionConfig->swapTol](ir::CircuitNode& circuit) {
          SizeOnlyFusionConfig cfoFusionConfig(2);
          cfoFusionConfig.swapTol = swapTol;
          cast::impl::applyCFOFusion(circuit, &cfoFusionConfig, 2);
        }
      );
    }
  }

  if (data->enableFusion) {
    auto pass =
      [fusionConfig=std::move(data->fusionConfig)](ir::CircuitNode& circuit) {
        cast::applyGateFusionPass(circuit, fusionConfig.get());
      };
    opt.addPass("Agglomerative Fusion Pass", std::move(pass));
  }
  
  return opt;
}