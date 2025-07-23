#include "cast/CPU/CPUOptimizer.h"

using namespace cast;

int main(int argc, char** argv) {
  CPUOptimizer opt;
  opt.setSizeOnlyFusionConfig(5);

  assert(argc > 1 && "Usage: scratch <qasm_file>");
  auto circuitOrErr = cast::parseCircuitFromQASMFile(argv[1]);
  if (!circuitOrErr) {
    std::cerr << "Failed to parse circuit: " << circuitOrErr.takeError() << std::endl;
    return 1;
  }
  auto circuit = circuitOrErr.takeValue();
  circuit.displayInfo(std::cerr << "Before Opt\n", 3);
  opt.run(circuit, utils::Logger(std::cerr, 1));
  circuit.displayInfo(std::cerr << "After Opt\n", 3);

  return 0;
}