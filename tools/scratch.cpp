#include "cast/QuantumChannel.h"

using namespace cast;

int main(int argc, char** argv) {

  std::cerr << "sizeof(ScalarGateMatrix) = " << sizeof(ScalarGateMatrix) << "\n";
  std::cerr << "sizeof(ComplexSquareMatrix) = " << sizeof(ComplexSquareMatrix) << "\n";
  std::cerr << "sizeof(KrausRep) = " << sizeof(KrausRep) << "\n";
  std::cerr << "sizeof(ChoiRep) = " << sizeof(ChoiRep) << "\n";
  
  KrausRep krausRep(1);
  krausRep.addMatrix(ScalarGateMatrix::X());
  krausRep.display(std::cerr);

  ChoiRep choiRep = ChoiRep::FromKrausRep(krausRep);
  choiRep.display(std::cerr);

  return 0;
}