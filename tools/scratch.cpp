#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "utils/Formats.h"
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
  km.enableTiming();

  QuantumGate::TargetQubitsType qubits;
  utils::sampleNoReplacement(nQubitsSV, 3, qubits);
  llvm::cantFail(km.genStandaloneGate(
      genCfg, StandardQuantumGate::RandomUnitary(qubits), "gateA"));

  utils::sampleNoReplacement(nQubitsSV, 4, qubits);
  llvm::cantFail(km.genStandaloneGate(
      genCfg, StandardQuantumGate::RandomUnitary(qubits), "gateB"));

  utils::sampleNoReplacement(nQubitsSV, 2, qubits);
  llvm::cantFail(km.genStandaloneGate(
      genCfg, StandardQuantumGate::RandomUnitary(qubits), "gateC"));

  std::vector<const CUDAKernelManager::ExecutionResult*> results;
  for (auto& kernel : km)
    results.push_back(km.enqueueKernelLaunch(kernel));

  km.syncKernelExecution();

  for (const auto* res : results) {
    std::cerr << "Kernel: " << res->kernelName << "\n"
              << "- Kernel Time: " << utils::fmt_time(res->getKernelTime())
              << "\n";
    std::cerr << "- Compile Time: "
              << utils::fmt_time(res->getCompileTime()) << "\n";
  }

  return 0;
}