#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "utils/PrintSpan.h"
#include "utils/utils.h"
#include <fstream>
#include <numeric>

using namespace cast;

int main(int argc, char** argv) {

  CUDAKernelGenConfig genCfg;
  CUDAKernelManager km(3);
  constexpr int nQubitsSV = 29;
  CUDAStatevectorF64 sv(nQubitsSV);
  sv.initialize();
  km.setLaunchConfig(sv.getDevicePtr(), nQubitsSV);

  int count = 0;
  for (int i = 0; i < 4; ++i) {
    QuantumGate::TargetQubitsType qubits;
    std::string name;
    QuantumGatePtr gate;
    utils::sampleNoReplacement(nQubitsSV, 2, qubits);
    gate = StandardQuantumGate::RandomUnitary(qubits);
    name = "g" + std::to_string(count++);
    if (auto k = km.genStandaloneGate(genCfg, gate, name); !k) {
      std::cerr << "Failed to generate kernel: "
                << llvm::toString(k.takeError()) << "\n";
      std::exit(1);
    }

    utils::sampleNoReplacement(nQubitsSV, 3, qubits);
    gate = StandardQuantumGate::RandomUnitary(qubits);
    name = "g" + std::to_string(count++);
    if (auto k = km.genStandaloneGate(genCfg, gate, name); !k) {
      std::cerr << "Failed to generate kernel: "
                << llvm::toString(k.takeError()) << "\n";
      std::exit(1);
    }
  }

  for (auto& kernel : km) {
    km.enqueueKernelLaunch(kernel);
    km.enqueueKernelLaunch(kernel);
  }

  km.syncKernelExecution(true);

  return 0;
}