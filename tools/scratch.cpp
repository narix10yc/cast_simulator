#include "cast/CPU/CPUKernelManager.h"
#include "openqasm/parser.h"
#include <iostream>

using namespace cast;

int main(int argc, char** argv) {
  assert(argc > 1);
  auto circuitOrError = parseCircuitFromQASMFile(argv[1]);
  if (!circuitOrError) {
    std::cerr << "Failed to parse circuit from QASM file: "
              << circuitOrError.takeError() << "\n";
    return 1;
  }
  auto circuit = circuitOrError.moveValue();
  circuit.print(std::cerr, 0);

  CPUKernelGenConfig config;
  CPUKernelManager kernelMgr;

  // Ignore errors
  // kernelMgr.genCPUGate(config, gate, "test_gate").consumeError();

  // kernelMgr.initJIT(1, llvm::OptimizationLevel::O0, false, 0).consumeError();




  return 0;
}