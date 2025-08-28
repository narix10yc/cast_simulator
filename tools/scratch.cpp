#include "cast/CUDA/CUDAKernelManager.h"

#include "utils/utils.h"

using namespace cast;

int main(int argc, char** argv) {
  CUDAKernelGenConfig config;
  CUDAKernelManager km(4);

  config.displayInfo(std::cerr);
  QuantumGate::TargetQubitsType qubits;
  for (int i = 0; i < 100; ++i) {
    utils::sampleNoReplacement(28, 4, qubits);
    km.genStandaloneGate(config,
                         StandardQuantumGate::RandomUnitary(qubits),
                         "gate_" + std::to_string(i))
        .consumeError();
  }

  utils::timedExecute([&]() { km.compileToPTX(1, 1); }, "Emit PTX");

  utils::timedExecute([&]() { km.compileToCUBIN(1, 1); }, "Init CUJIT");

  km.getKernelByName("gate_10")->displayInfo(std::cerr);

  km.displayInfo(std::cerr);
}