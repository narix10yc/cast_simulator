#include "cast/CPU/CPUOptimizerBuilder.h"

using namespace cast;

int main(int argc, char** argv) {
  CPUOptimizerBuilder builder;
  builder
        //  .disableCanonicalization()
         .disableCFO()
         .setSizeOnlyFusion(7)
        //  .setNThreads(1)
        //  .setPrecision(Precision::F64)
        //  .setZeroTol(0.0)
        //  .setSwapTol(0.0)
        //  .setVerbose(1);
        ;

  auto optOrErr = builder.build();
  if (!optOrErr) {
    std::cerr << "Failed to build optimizer: " << optOrErr.takeError() << std::endl;
    return 1;
  }
  auto opt = optOrErr.takeValue();

  assert(argc > 1 && "Usage: scratch <qasm_file>");
  auto circuitOrErr = cast::parseCircuitFromQASMFile(argv[1]);
  if (!circuitOrErr) {
    std::cerr << "Failed to parse circuit: " << circuitOrErr.takeError() << std::endl;
    return 1;
  }
  auto circuit = circuitOrErr.takeValue();
  circuit.displayInfo(std::cerr << "Before Opt\n", 3);
  opt.run(circuit, 1);
  circuit.displayInfo(std::cerr << "After Opt\n", 3);

  return 0;
}