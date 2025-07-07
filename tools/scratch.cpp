#include "cast/Core/Optimize.h"
#include "utils/utils.h"

using namespace cast;

int main(int argc, char** argv) {
  assert(argc > 1 && "No arguments provided");

  auto circuitOrErr = cast::parseCircuitFromQASMFile(argv[1]);
  if (!circuitOrErr) {
    std::cerr << "Failed to parse circuit from file: " << argv[1]
              << "\nError: " << circuitOrErr.takeError() << "\n";
    return 1;
  }
  auto circuit = circuitOrErr.moveValue();
  circuit.displayInfo(std::cerr << "Initial Circuit Info\n", 3);
  // circuit.visualize(std::cerr << "Initial Circuit Visualization\n");

  cast::applyCanonicalizationPass(circuit, 1e-8);
  circuit.displayInfo(std::cerr << "\nAfter Canonicalization\n", 3);
  // circuit.visualize(std::cerr << "\nAfter Canonicalization Visualization\n");

  // auto fusionConfig = cast::FusionConfig::SizeOnly(3);
  cast::FusionConfig fusionConfig;
  fusionConfig.costModel = std::make_unique<cast::SizeOnlyCostModel>(5, -1, 1e-8); // Use size-only fusion
  fusionConfig.precision = cast::Precision::F64; // Use double precision
  fusionConfig.nThreads = 1; // Single-threaded for this example
  fusionConfig.zeroTol = 1e-8; // Tolerance for zero gates
  fusionConfig.swaTol = 0; // Disable swapping for this example
  fusionConfig.maxKOverride = cast::GLOBAL_MAX_K; // Use global max k
  fusionConfig.benefitMargin = 0.0;
  fusionConfig.incrementScheme = true; // Use incremental scheme
  // fusionConfig.multiTraversal = false;
  fusionConfig.multiTraversal = true; // Enable multi-traversal for better optimization

  utils::timedExecute(
    [&]() {
      cast::applyGateFusionPass(circuit, fusionConfig, false);
    },
    "Gate Fusion Pass");
  circuit.displayInfo(std::cerr << "\nAfter Fusion\n", 3);
  // circuit.visualize(std::cerr << "\nAfter Fusion Visualization\n");

  return 0;
}