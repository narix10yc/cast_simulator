#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "utils/PrintSpan.h"
#include "utils/utils.h"
#include <fstream>
#include <numeric>

using namespace cast;

int main(int argc, char** argv) {

  CUDAKernelGenConfig genCfg;
  CUDAKernelManager km(2);
  constexpr int nQubitsSV = 28;
  CUDAStatevectorF64 sv(nQubitsSV);
  sv.initialize();
  km.setLaunchConfig(sv.getDevicePtr(), nQubitsSV);

  for (int i = 0; i < 10; ++i) {
    QuantumGate::TargetQubitsType qubits;
    utils::sampleNoReplacement(nQubitsSV, 2, qubits);
    auto gate = StandardQuantumGate::RandomUnitary(qubits);
    auto name = "g" + std::to_string(i);
    if (auto k = km.genStandaloneGate(genCfg, gate, name); !k) {
      std::cerr << "Failed to generate kernel: "
                << llvm::toString(k.takeError()) << "\n";
      std::exit(1);
    }
  }

  for (auto& kernel : km) {
    km.enqueueKernelLaunch(kernel, 999);
  }

  km.syncKernelExecution();
  km.syncKernelExecution();

  return 0;
}