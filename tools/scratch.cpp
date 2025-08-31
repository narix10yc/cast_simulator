#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"

#include "cuda.h"

#include "utils/utils.h"

using namespace cast;

int main(int argc, char** argv) {
  constexpr int nQubits = 28;

  CUDAStatevectorF64 sv(nQubits);
  sv.initialize();
  CUDAKernelGenConfig config;
  CUDAKernelManager km(2);

  for (int i = 0; i < 10; ++i) {
    QuantumGate::TargetQubitsType qubits;
    utils::sampleNoReplacement(nQubits, 2, qubits);
    km.genStandaloneGate(config,
                         StandardQuantumGate::RandomUnitary(qubits),
                         "gate_" + std::to_string(i))
        .consumeError();
  }

  km.setLaunchConfig(sv.getDevicePtr(), nQubits);
  for (auto& kernel : km)
    km.enqueueKernelLaunch(kernel);

  std::cerr << "Main thread: all kernel launches enqueued\n";
  km.syncKernelExecution();
  std::cerr << "Main thread Round 1: primary stream synced\n";

  for (auto& kernel : km) {
    km.enqueueKernelLaunch(kernel);
    km.syncKernelExecution();
  }

  return 0;
}