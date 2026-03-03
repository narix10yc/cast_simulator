#include "cast/Core/QuantumGate.h"
#include "cast/FPGA/FPGAGateCategory.h"
#include "tests/TestKit.h"

using namespace cast;

bool test::test_fpgaGateCategory() {
  TestSuite suite("FPGA gate categorization rules");

  auto hCate = fpga::getFPGAGateCategory(StandardQuantumGate::H(0).get());
  suite.assertEqual(hCate.is(fpga::FPGAGateCategory::SingleQubit),
                    true,
                    "Hadamard is classified as SingleQubit",
                    GET_INFO());
  suite.assertEqual(hCate.is(fpga::FPGAGateCategory::UnitaryPerm),
                    false,
                    "Hadamard is not classified as UnitaryPerm",
                    GET_INFO());
  suite.assertEqual(hCate.is(fpga::FPGAGateCategory::NonComp),
                    false,
                    "Hadamard is not classified as NonComp",
                    GET_INFO());
  suite.assertEqual(hCate.is(fpga::FPGAGateCategory::RealOnly),
                    true,
                    "Hadamard is classified as RealOnly",
                    GET_INFO());

  auto i1Cate = fpga::getFPGAGateCategory(StandardQuantumGate::I1(0).get());
  suite.assertEqual(i1Cate.is(fpga::FPGAGateCategory::SingleQubit),
                    true,
                    "Identity is classified as SingleQubit",
                    GET_INFO());
  suite.assertEqual(i1Cate.is(fpga::FPGAGateCategory::UnitaryPerm),
                    true,
                    "Identity is classified as UnitaryPerm",
                    GET_INFO());
  suite.assertEqual(i1Cate.is(fpga::FPGAGateCategory::NonComp),
                    true,
                    "Identity is classified as NonComp",
                    GET_INFO());
  suite.assertEqual(i1Cate.is(fpga::FPGAGateCategory::RealOnly),
                    true,
                    "Identity is classified as RealOnly",
                    GET_INFO());

  // Display results
  return suite.displayResult();
}
