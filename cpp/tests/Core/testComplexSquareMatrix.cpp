#include "cast/ADT/ComplexSquareMatrix.h"
#include "tests/TestKit.h"

using namespace cast;

bool test::test_complexSquareMatrix() {
  test::TestSuite suite("ComplexSquareMatrix algebra and inversion");

  auto matI = ComplexSquareMatrix::I1();
  auto matX = ComplexSquareMatrix::X();
  auto matY = ComplexSquareMatrix::Y();
  auto matZ = ComplexSquareMatrix::Z();
  auto matResult = ComplexSquareMatrix(2);

  cast::matmul(matX, matX, matResult);
  suite.assertCloseFP64(
      maximum_norm(matResult, matI), 0.0, "X * X equals identity", GET_INFO());

  cast::matmul(matY, matY, matResult);
  suite.assertCloseFP64(
      maximum_norm(matResult, matI), 0.0, "Y * Y equals identity", GET_INFO());

  cast::matmul(matZ, matZ, matResult);
  suite.assertCloseFP64(
      maximum_norm(matResult, matI), 0.0, "Z * Z equals identity", GET_INFO());

  auto mat_iZ = matZ * std::complex<double>(0, 1);
  cast::matmul(matX, matY, matResult);
  suite.assertCloseFP64(
      maximum_norm(matResult, mat_iZ), 0.0, "X * Y equals iZ", GET_INFO());

  auto matA = ComplexSquareMatrix::RandomUnitary(4);
  ComplexSquareMatrix matAInv;
  auto rst = cast::matinv(matA, matAInv);
  if (!rst) {
    suite.assertEqual(
        1, 0, "Matrix inversion succeeds for random unitary", GET_INFO());
  } else {
    ComplexSquareMatrix matAAInv;
    cast::matmul(matA, matAInv, matAAInv);
    suite.assertCloseFP64(maximum_norm(matAAInv, ComplexSquareMatrix::eye(4)),
                          0.0,
                          "A * inverse(A) equals identity",
                          GET_INFO());
  }

  return suite.displayResult();
}
