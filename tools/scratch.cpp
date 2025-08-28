#include "cast/CUDA/CUDAKernelManager.h"

#include "utils/utils.h"

using namespace cast;

int main(int argc, char** argv) {
  CUDAKernelGenConfig config;
  CUDAKernelManager km(10);

  config.displayInfo(std::cerr);
  QuantumGate::TargetQubitsType qubits;
  for (int i = 0; i < 10; ++i) {
    utils::sampleNoReplacement(28, 5, qubits);
    km.genStandaloneGate(config,
                         StandardQuantumGate::RandomUnitary(qubits),
                         "gate_" + std::to_string(i))
        .consumeError();
  }

  utils::timedExecute(
      [&]() {
        auto r = km.initJIT(1, 1);
        if (!r) {
          std::cerr << "Failed to initialize JIT: " << r.takeError() << "\n";
        }
      },
      "Initialize JIT");

  km.getKernelByName("gate_3")->displayInfo(std::cerr);

  km.displayInfo(std::cerr);
}