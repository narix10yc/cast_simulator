#include "cast/Core/QuantumGate.h"

using namespace cast;

int main(int argc, char** argv) {

  std::cerr << "sizeof(std::unique_ptr<int[]>) = "
            << sizeof(std::unique_ptr<int[]>) << "\n";

  ComplexSquareMatrix A{
    // real part
    {1, 2, 3, 4},
    // imag part
    {0, 0, 0, 0}
  };
  ComplexSquareMatrix AInv;
  if (!cast::matinv(A, AInv)) {
    std::cerr << "Matrix inversion failed.\n";
    return 1;
  }
  A.print(std::cerr) << "\n";
  AInv.print(std::cerr) << "\n";
  ComplexSquareMatrix AAInv;
  cast::matmul(A, AInv, AAInv);
  AAInv.print(std::cerr) << "\n";
  
  auto Id = ComplexSquareMatrix::eye(2);
  std::cerr << "Maximum norm: "
            << cast::maximum_norm(AAInv, Id) << "\n";
}