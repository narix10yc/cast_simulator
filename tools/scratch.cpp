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

  utils::timedExecute([&]() { km.compileLLVMIRToPTX(1, 1); }, "Compile to PTX");

  utils::timedExecute([&]() { km.compilePTXToCubin(1, 1); }, "Compile to CUBIN");

  utils::timedExecute([&]() { km.loadCubin(1); }, "Load Cubin");

  km.getKernelByName("gate_3")->displayInfo(std::cerr);

  km.displayInfo(std::cerr);
}