#include "cast/Core/QuantumGate.h"

using namespace cast;

int main(int argc, char** argv) {
  std::cerr << "sizeof(ScalarGateMatrix) = " << sizeof(ScalarGateMatrix) << "\n";
  std::cerr << "sizeof(ComplexSquareMatrix) = " << sizeof(ComplexSquareMatrix) << "\n";
  std::cerr << "sizeof(KrausRep) = " << sizeof(KrausRep) << "\n";
  std::cerr << "sizeof(ChoiRep) = " << sizeof(ChoiRep) << "\n";

  auto quantumGate = StandardQuantumGate::Create(
    ScalarGateMatrix::I1(), NoiseChannel::SymmetricPauliChannel(0.03), {0});

  quantumGate->displayInfo(std::cerr, 3);

  auto superopGate = cast::getSuperopGate(quantumGate);
  superopGate->displayInfo(std::cerr, 3);

  return 0;
}