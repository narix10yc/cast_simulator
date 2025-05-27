#include "tests/TestKit.h"
#include "cast/ADT/ComplexSquareMatrix.h"

using namespace cast;

void test::test_complexSquareMatrix() {
  test::TestSuite suite("Complex Square Matrix Test");

  auto matI = ComplexSquareMatrix::I1();
  auto matX = ComplexSquareMatrix::X();
  auto matY = ComplexSquareMatrix::Y();
  auto matZ = ComplexSquareMatrix::Z();

  suite.assertClose(maximum_norm(matX.matmul(matX), matI), 0.0,
                    "XX == I", GET_INFO());
  suite.assertClose(maximum_norm(matY.matmul(matY), matI), 0.0,
                    "YY == I", GET_INFO());
  suite.assertClose(maximum_norm(matZ.matmul(matZ), matI), 0.0,
                    "ZZ == I", GET_INFO());
                    
  auto mat_iZ = matZ * std::complex<double>(0, 1);
  suite.assertClose(maximum_norm(matX.matmul(matY), mat_iZ), 0.0,
                    "XY == iZ", GET_INFO());

  suite.displayResult();
}