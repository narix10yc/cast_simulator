#include "cast/Core/Optimize.h"
#include "cast/CPU/CPUFusionConfig.h"
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

  utils::timedExecute(
    [&]() {
      cast::applyCanonicalizationPass(circuit, 1e-8);
    },
    "Canonicalization Pass"
  );
  circuit.displayInfo(std::cerr << "\nAfter Canonicalization\n", 3);
  // circuit.visualize(std::cerr << "\nAfter Canonicalization Visualization\n");

  cast::CPUFusionConfig cpuFusionConfig;

  utils::timedExecute(
    [&]() {
      cast::applyGateFusionPass(circuit, &cpuFusionConfig);
    },
    "Gate Fusion Pass"
  );
  circuit.displayInfo(std::cerr << "\nAfter Fusion\n", 3);
  // circuit.visualize(std::cerr << "\nAfter Fusion Visualization\n");

  return 0;
}