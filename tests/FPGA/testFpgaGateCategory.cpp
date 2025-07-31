#include "cast/Core/QuantumGate.h"
#include "cast/FPGA/FPGAGateCategory.h"
#include "tests/TestKit.h"

using namespace cast;

void test::test_fpgaGateCategory() {
  TestSuite suite("FPGA Gate Category Test");

  auto hCate = fpga::getFPGAGateCategory(StandardQuantumGate::H(0).get());
  suite.assertEqual(hCate.is(fpga::FPGAGateCategory::SingleQubit),
                    true,
                    "Hadamard Gate is single-qubit",
                    GET_INFO());
  suite.assertEqual(hCate.is(fpga::FPGAGateCategory::UnitaryPerm),
                    false,
                    "Hadamard Gate is not Unitary Perm",
                    GET_INFO());
  suite.assertEqual(hCate.is(fpga::FPGAGateCategory::NonComp),
                    false,
                    "Hadamard Gate is not Non-Computational",
                    GET_INFO());
  suite.assertEqual(hCate.is(fpga::FPGAGateCategory::RealOnly),
                    true,
                    "Hadamard Gate is Real Only",
                    GET_INFO());

  auto i1Cate = fpga::getFPGAGateCategory(StandardQuantumGate::I1(0).get());
  suite.assertEqual(i1Cate.is(fpga::FPGAGateCategory::SingleQubit),
                    true,
                    "Identity Gate is single-qubit",
                    GET_INFO());
  suite.assertEqual(i1Cate.is(fpga::FPGAGateCategory::UnitaryPerm),
                    true,
                    "Identity Gate is Unitary Perm",
                    GET_INFO());
  suite.assertEqual(i1Cate.is(fpga::FPGAGateCategory::NonComp),
                    true,
                    "Identity Gate is Non-Computational",
                    GET_INFO());
  suite.assertEqual(i1Cate.is(fpga::FPGAGateCategory::RealOnly),
                    true,
                    "Identity Gate is Real Only",
                    GET_INFO());

  // Display results
  suite.displayResult();
}