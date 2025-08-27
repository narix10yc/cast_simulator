#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAOptimizer.h"

#include "cast/CPU/CPUOptimizer.h"
#include "cast/CUDA/CUDAStatevector.h"

#include "utils/utils.h"

using namespace cast;

int main(int argc, char** argv) {
  assert(argc > 1);
  std::string fileName(argv[1]);

  auto cudaPerfCache = std::make_unique<CUDAPerformanceCache>();
  cudaPerfCache->loadFromFile("rtx5080.csv");
  auto cudaCM = std::make_unique<CUDAAdvCostModel>(std::move(cudaPerfCache));
  cudaCM->displayInfo(std::cerr, 3);

  auto cudaFusionConfig = std::make_unique<CUDAFusionConfig>(std::move(cudaCM));
  cudaFusionConfig->setPrecision(Precision::F64);

  CUDAOptimizer cudaOpt;
  cudaOpt.setCUDAFusionConfig(std::move(cudaFusionConfig));

  auto circuitOrErr = cast::parseCircuitFromQASMFile(fileName);
  if (!circuitOrErr) {
    std::cerr << "Error parsing QASM file: " << circuitOrErr.takeError()
              << "\n";
    return 1;
  }

  auto circuit = circuitOrErr.takeValue();
  circuit.displayInfo(std::cerr << "Before Opt:\n");

  cudaOpt.fusionConfig = std::make_unique<SizeOnlyFusionConfig>(3);
  cudaOpt.run(circuit, {std::cerr, 2});
  circuit.displayInfo(std::cerr << "After Opt:\n");

  CUDAKernelManager cudaKernelMgr;
  CUDAKernelGenConfig cudaGenConfig;
  cudaGenConfig.displayInfo(std::cerr);

  cudaKernelMgr
      .genGraphGates(
          cudaGenConfig, *circuit.getAllCircuitGraphs()[0], "myGraph")
      .consumeError();

  cudaKernelMgr.emitPTX(llvm::OptimizationLevel::O3, 1);
  cudaKernelMgr.initCUJIT(1);

  CUDAStatevectorF64 sv(28);
  sv.initialize();

  auto kernels = cudaKernelMgr.getKernelsFromGraphName("myGraph");
  for (const auto& kernel : kernels) {
    utils::timedExecute([&]() {
      cudaKernelMgr.launchCUDAKernel(sv.dData(), sv.nQubits(), *kernel, 0);
    }, kernel->llvmFuncName.c_str());
  }

  return 0;
}