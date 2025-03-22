#include "tests/TestKit.h"
#include "utils/utils.h"

using namespace cast::test;

int main() {
  utils::timedExecute([] {
    test_applyGate();
    test_gateMatMul();
  }, "Gate Multiplication Test Finished!");

  utils::timedExecute([] {
    test_cpuH();
    test_cpuU();
  }, "CPU Codegen Test Finished!");

  utils::timedExecute([] {
    test_fusionCPU();
  }, "CPU Fusion Test Finished!");

  utils::timedExecute([] {
    test_cpuRz_param();
  }, "CPU Codegen (Runtime) Test Finished!");

  #ifdef CAST_USE_CUDA

  utils::timedExecute([] {
    test_statevectorCUDA();
  }, "StatevectorCUDA Test Finished!");

  utils::timedExecute([] {
    test_cudaU();
  }, "CUDA Codegen Test Finished!");

  #endif // CAST_USE_CUDA

  
  return 0;
}