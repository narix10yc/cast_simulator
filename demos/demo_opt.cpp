#include "cast/CPU/CPUOptimizer.h"

using namespace cast;

int main(int argc, char** argv) {
  assert(argc > 1 && "No input file specified");

  auto circuitOrErr = cast::parseCircuitFromQASMFile(argv[1]);
  if (!circuitOrErr) {
    std::cerr << BOLDRED("[Error]: ") << "Failed to parse QASM file.\n";
    return 1;
  }
  auto circuitOriginal = circuitOrErr.takeValue();
  
  utils::Logger logger(std::cerr, 3);
  
  for (int size = 3; size <= 6; ++size) {
    auto circuit = circuitOriginal.copy();
    CPUOptimizer opt;
    opt.setSizeOnlyFusionConfig(size)
       .disableCFO();
    logger.log() << "Running size-only fusion with size = " << size << "\n";
    opt.run(circuit, logger);
  }

  return 0;
}