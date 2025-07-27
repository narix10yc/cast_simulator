#include "cast/ADT/ComplexSquareMatrix.h"
#include "tests/TestKit.h"

using namespace cast;

void test::test_complexSquareMatrix() {
  test::TestSuite suite("Complex Square Matrix Test");

  auto matI = ComplexSquareMatrix::I1();
  auto matX = ComplexSquareMatrix::X();
  auto matY = ComplexSquareMatrix::Y();
  auto matZ = ComplexSquareMatrix::Z();
  auto matResult = ComplexSquareMatrix(2);

  cast::matmul(matX, matX, matResult);
  suite.assertCloseF64(maximum_norm(matResult, matI), 0.0, "XX == I", GET_INFO());

  cast::matmul(matY, matY, matResult);
  suite.assertCloseF64(maximum_norm(matResult, matI), 0.0, "YY == I", GET_INFO());

  cast::matmul(matZ, matZ, matResult);
  suite.assertCloseF64(maximum_norm(matResult, matI), 0.0, "ZZ == I", GET_INFO());

  auto mat_iZ = matZ * std::complex<double>(0, 1);
  cast::matmul(matX, matY, matResult);
  suite.assertCloseF64(
      maximum_norm(matResult, mat_iZ), 0.0, "XY == iZ", GET_INFO());

  auto matA = ComplexSquareMatrix::RandomUnitary(4);
  ComplexSquareMatrix matAInv;
  auto rst = cast::matinv(matA, matAInv);
  if (!rst) {
    suite.assertEqual(1, 0, "Matrix inversion failed", GET_INFO());
  } else {
    ComplexSquareMatrix matAAInv;
    cast::matmul(matA, matAInv, matAAInv);
    suite.assertCloseF64(maximum_norm(matAAInv, ComplexSquareMatrix::eye(4)),
                      0.0,
                      "A * AInv == I",
                      GET_INFO());
  }

  suite.displayResult();
}