#include "cast/QuantumGate.h"

using namespace cast;

int main(int argc, char** argv) {

  std::cerr << "sizeof(ScalarGateMatrix) = " << sizeof(ScalarGateMatrix) << "\n";
  std::cerr << "sizeof(ComplexSquareMatrix) = " << sizeof(ComplexSquareMatrix) << "\n";
  std::cerr << "sizeof(KrausRep) = " << sizeof(KrausRep) << "\n";
  std::cerr << "sizeof(ChoiRep) = " << sizeof(ChoiRep) << "\n";
  
  auto gate = QuantumGate::Create(ScalarGateMatrix::X());
  gate->setNoiseSymmetricPauliChannel(0.1);

  gate->displayInfo(std::cerr, 3);

  return 0;
}