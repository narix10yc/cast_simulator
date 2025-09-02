#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"

#include "utils/utils.h"
#include <random>

using namespace cast;

constexpr int nQubits = 29;

int main(int argc, char** argv) {

  CUDAKernelManager km(12);
  CUDAKernelGenConfig config;

  for (int i = 0; i < 1000; ++i) {
    std::string name = "gate_" + std::to_string(i);
    QuantumGate::TargetQubitsType qubits;
    std::random_device rd;
    std::uniform_int_distribution<int> dist(1, 4);
    utils::sampleNoReplacement(nQubits, dist(rd), qubits);
    km.genStandaloneGate(
          config, StandardQuantumGate::RandomUnitary(qubits), name)
        .consumeError();
  }

  CUDAStatevectorF64 sv(nQubits);
  sv.initialize();
  km.setLaunchConfig(sv.getDevicePtr(), nQubits);

  const CUDAKernelManager::ExecutionResult* lastKernelLaunchInfo = nullptr;
  for (auto& kernel : km)
    lastKernelLaunchInfo = km.enqueueKernelLaunch(kernel, 2);

  km.syncKernelExecution(true);
  lastKernelLaunchInfo->displayInfo(std::cerr);

  return 0;
}