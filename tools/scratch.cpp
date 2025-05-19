#include "cast/QuantumGate.h"

using namespace cast;

int main(int argc, char** argv) {
  std::cerr << "sizeof(ScalarGateMatrix) = " << sizeof(ScalarGateMatrix) << "\n";
  std::cerr << "sizeof(ComplexSquareMatrix) = " << sizeof(ComplexSquareMatrix) << "\n";
  std::cerr << "sizeof(KrausRep) = " << sizeof(KrausRep) << "\n";
  std::cerr << "sizeof(ChoiRep) = " << sizeof(ChoiRep) << "\n";
  
  auto xGate = ScalarGateMatrix::X();
  auto pauliError = NoiseChannel::SymmetricPauliChannel(0.1);

  auto quantumGate = QuantumGate::Create(xGate, pauliError, {0});

  quantumGate->displayInfo(std::cerr, 3);

  return 0;
}