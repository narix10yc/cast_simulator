#include "cast/CPU/CPUKernelManager.h"

using namespace cast;

MaybeError<int> f() {
  return cast::makeError<int>("some msg");
}

MaybeError<int> g() {
  return 42;
}

int main(int argc, char** argv) {
  auto gate = StandardQuantumGate::RandomUnitary({3, 1});

  CPUKernelGenConfig config;
  CPUKernelManager kernelMgr;

  auto result = f();
  if (!result) {
    std::cerr << "Error: " << result.takeError() << "\n";
  }

  result = g();
  if (!result) {
    std::cerr << "Error: " << result.takeError() << "\n";
  } else {
    std::cout << "Result: " << *result << "\n";
  }
  
  return 0;
}