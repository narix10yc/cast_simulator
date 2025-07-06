#include "cast/Core/Optimize.h"

int main(int argc, char** argv) {
  assert(argc > 1 && "No arguments provided");

  auto circuitOrErr = cast::parseCircuitFromQASMFile(argv[1]);
  if (!circuitOrErr) {
    std::cerr << "Failed to parse circuit from file: " << argv[1]
              << "\nError: " << circuitOrErr.takeError() << "\n";
    return 1;
  }
  auto circuit = circuitOrErr.moveValue();

  cast::applyCanonicalizationPass(circuit);

  circuit.displayInfo(std::cerr << "\nAfter Canonicalization\n", 3);

  cast::applyGateFusionPass(circuit, cast::FusionConfig::SizeOnly(7), false);

  circuit.displayInfo(std::cerr << "\nAfter Fusion\n", 3);

  return 0;
}