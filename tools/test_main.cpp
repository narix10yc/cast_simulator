#include "tests/TestKit.h"
#include "utils/utils.h"

using namespace cast::test;

int main() {
  utils::timedExecute([] { test_complexSquareMatrix(); },
                      "Complex Square Matrix Test Finished!");

  utils::timedExecute(
      [] {
        test_cpuH();
        test_cpuU();
      },
      "CPU Codegen Test Finished!");

  utils::timedExecute([] { test_quantumGate(); },
                      "Quantum Gate Test Finished!");

  utils::timedExecute([] { test_quantumChannel(); },
                      "Quantum Channel Test Finished!");

  utils::timedExecute([] { test_fusionCPU(); }, "CPU Fusion Test Finished!");

  // utils::timedExecute([] {
  //   test_fpgaGateCategory();
  // }, "FPGA Gate Category Test Finished!");

  // utils::timedExecute([] {
  //   test_cpuRz_param();
  // }, "CPU Codegen (Runtime) Test Finished!");

#ifdef CAST_USE_CUDA

  utils::timedExecute([] { test_statevectorCUDA(); },
                      "CUDAStatevector Test Finished!");

  utils::timedExecute([] { test_cudaU(); }, "CUDA Codegen Test Finished!");

  utils::timedExecute([] { test_cudaU2(); }, "CUDA Codegen Test Finished!");

  utils::timedExecute([] { test_cuda_gate_var(); },
                      "CUDA Codegen Test Finished!");

  // utils::timedExecute([] { test_cudaRz_param(); },
  // "CUDA Codegen (Runtime) Test Finished!");

#endif // CAST_USE_CUDA

  return 0;
}