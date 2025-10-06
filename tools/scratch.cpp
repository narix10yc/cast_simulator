#include "cast/CPU/CPUOptimizer.h"

static std::ostream& logerr() { return std::cerr << BOLDRED("[Error]: "); }

int main(int argc, char** argv) {
  assert(argc == 2 && "Usage: demo_scratch <qasm-file>");
  cast::CPUOptimizer opt;
  opt.enableCFO(false).enableFusion(false);

  auto circuit = cast::parseCircuitFromQASMFile(argv[1]);
  if (!circuit) {
    logerr() << "Failed to parse QASM file: "
             << llvm::toString(circuit.takeError()) << "\n";
    return 1;
  }

  opt.run(**circuit, utils::Logger(std::cerr, 1));

  auto allGates = (*circuit)->getAllCircuitGraphs()[0]->getAllGates();
  int nTwoQubitGates = 0;
  double opCount = 0.0;
  double opCountDense = 0.0;
  for (const auto& g : allGates) {
    if (g->nQubits() == 2)
      nTwoQubitGates++;
    opCount += g->opCount(1e-8);
    opCountDense += g->opCount(0.0);
  }

  std::cout << "After optimization:\n"
            << "- Total gates:     " << allGates.size() << "\n"
            << "- Two-qubit gates: " << nTwoQubitGates << "\n"
            << "- Sparsity Metric: " << opCount / opCountDense << " ("
            << opCount << " / " << opCountDense << ")\n";

  return 0;
}