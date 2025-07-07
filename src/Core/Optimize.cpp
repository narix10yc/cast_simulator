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
  .costModel = nullptr,
  .precision = Precision::Unknown,
  .zeroTol = 1e-8,
  .swaTol = 0.0,
  .incrementScheme = true,
  .multiTraversal = false,
  .maxKOverride = 3,
  .benefitMargin = 0.2,
};

const FusionConfig FusionConfig::Default {
  .costModel = nullptr,
  .precision = Precision::Unknown,
  .zeroTol = 1e-8,
  .swaTol = 0.0,
  .incrementScheme = true,
  .multiTraversal = false,
  .maxKOverride = 5,
  .benefitMargin = 0.0,
};

const FusionConfig FusionConfig::Aggressive {
  .costModel = nullptr,
  .precision = Precision::Unknown,
  .zeroTol = 1e-8,
  .swaTol = 1e-8,
  .incrementScheme = true,
  .multiTraversal = true,
  .maxKOverride = GLOBAL_MAX_K,
  .benefitMargin = 0.0,
};

FusionConfig FusionConfig::SizeOnly(int max_k) {
  return FusionConfig {
    .costModel = nullptr,
    .precision = Precision::Unknown,
    .zeroTol = 1e-8,
    .swaTol = 1e-8,
    .incrementScheme = true,
    .multiTraversal = false,
    .maxKOverride = max_k,
    .benefitMargin = 0.0,
  };
}

void cast::applyGateFusionPass(ir::CircuitNode& circuit,
                               const FusionConfig& _config,
                               bool applyCFO) {
  auto allCircuitGraphs = circuit.getAllCircuitGraphs();
  for (auto* graph : allCircuitGraphs)
    impl::applyGateFusion(*graph, _config);
}

void cast::applyCanonicalizationPass(ir::CircuitNode& circuit, double swaTol) {
  // perform fusion in each block
  auto allCircuitGraphs = circuit.getAllCircuitGraphs();
  for (auto* graph : allCircuitGraphs)
    cast::impl::applySizeTwoFusion(*graph, swaTol);

  circuit.displayInfo(std::cerr << "\nAfter Size-2 Fusion\n", 1);
  circuit.visualize(std::cerr) << "\n";

  auto cfoFusionConfig = FusionConfig::SizeOnly(2);
  cfoFusionConfig.swaTol = swaTol;
  cast::impl::applyCFOFusion(circuit, cfoFusionConfig, 2);
}