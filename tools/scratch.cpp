#include "cast/Core/QuantumGate.h"

using namespace cast;

int main(int argc, char** argv) {

  auto gate0 = StandardQuantumGate::H(0);
  auto gate1 = StandardQuantumGate::I1(1);

  auto gate = cast::matmul(gate0.get(), gate1.get());
  gate->displayInfo(std::cerr, 3);

}