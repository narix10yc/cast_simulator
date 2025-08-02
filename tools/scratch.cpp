#include "cast/CPU/CPUStatevector.h"
#include "cast/CPU/CPUKernelManager.h"

using namespace cast;

int main(int argc, char** argv) {

  QuantumGatePtr qGate = StandardQuantumGate::S(0);
  qGate = cast::matmul(qGate.get(), StandardQuantumGate::S(1).get());

  qGate->displayInfo(std::cout, 2);

  return 0;
}