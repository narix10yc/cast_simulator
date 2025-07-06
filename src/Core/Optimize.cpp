#include "cast/Core/Optimize.h"
#include "cast/Core/ImplOptimize.h"
#include "utils/iocolor.h"
#include "utils/PrintSpan.h"

#include "llvm/Support/Casting.h"

#define DEBUG_TYPE "fusion-new"
#include "llvm/Support/Debug.h"
// #define LLVM_DEBUG(X) X

using namespace cast;

const FusionConfig FusionConfig::Minor {
  .precision = Precision::Unknown,
  .zeroTol = 1e-8,
  .incrementScheme = true,
  .maxKOverride = 3,
  .benefitMargin = 0.2,
};

const FusionConfig FusionConfig::Default {
  .precision = Precision::Unknown,
  .zeroTol = 1e-8,
  .incrementScheme = true,
  .maxKOverride=6,
  .benefitMargin = 0.0,
};

const FusionConfig FusionConfig::Aggressive {
  .precision = Precision::Unknown,
  .zeroTol = 1e-8,
  .incrementScheme = true,
  .maxKOverride = GLOBAL_MAX_K,
  .benefitMargin = 0.0,
};

FusionConfig FusionConfig::SizeOnly(int max_k) {
  return FusionConfig {
    .costModel = nullptr,
    .precision = Precision::Unknown,
    .zeroTol = 1e-8,
    .incrementScheme = false,
    .maxKOverride = max_k,
    .benefitMargin = 0.0,
  };
}

void cast::applyGateFusion(ir::CircuitGraphNode& graph,
                           const FusionConfig& config) {
  int curMaxK = (config.incrementScheme ? 2 : config.maxKOverride);
  do {
    auto it = graph.tile_begin();
    // we need to query graph.tile_end() every time, because impl::startFusion may
    // change graph tile
    while (it != graph.tile_end()) {
      for (int q = 0; q < graph.nQubits(); ++q) 
        cast::impl::startFusion(graph, config, curMaxK, it, q);
      ++it;
    }
    graph.squeeze();
  } while (++curMaxK <= config.maxKOverride);
}


void cast::applyGateFusionPass(ir::CircuitNode& circuit,
                               const FusionConfig& _config,
                               bool applyCFO) {
  auto allCircuitGraphs = circuit.getAllCircuitGraphs();
  if (_config.incrementScheme == false) {
    for (auto* graph : allCircuitGraphs)
      applyGateFusion(*graph, _config);
    return;
  }

  // temporary work-around. manual control of increment scheme
  auto config = _config;
  config.incrementScheme = false;
  for (int curMaxK = 2; curMaxK <= config.maxKOverride; ++curMaxK) {
    config.maxKOverride = curMaxK;
    for (auto* graph : allCircuitGraphs)
      applyGateFusion(*graph, config);
    if (applyCFO) {
      SizeOnlyCostModel fusionCFOCostModel(curMaxK, -1, config.zeroTol);
      impl::applyFusionCFOPass(circuit, config, &fusionCFOCostModel, curMaxK);
    }
  }
}

void cast::applyCanonicalizationPass(ir::CircuitNode& circuit, double swaTol) {
  // perform fusion in each block
  auto allCircuitGraphs = circuit.getAllCircuitGraphs();
  for (auto* graph : allCircuitGraphs)
    cast::impl::applySizeOnlyFusion(*graph, 2, swaTol);

}