#include "cast/Core/QuantumGate.h"
#include "tests/TestKit.h"
#include "llvm/Support/Casting.h"

using namespace cast;
using namespace cast::test;

void cast::test::test_quantumGate() {
  TestSuite suite("QuantumGate Test Suite");

  StandardQuantumGatePtr gate0, gate1, gate;
  const auto check = [&gate0, &gate1, &gate, &suite](const std::string& title,
                                                     const std::string& info) {
    auto prod = cast::matmul(gate0.get(), gate1.get());
    auto* stdQuGate = llvm::dyn_cast<StandardQuantumGate>(prod.get());
    assert(stdQuGate != nullptr);
    assert(stdQuGate->getScalarGM() != nullptr);
    assert(gate0->getScalarGM() != nullptr);
    assert(gate1->getScalarGM() != nullptr);
    assert(gate->getScalarGM() != nullptr);
    suite.assertClose(cast::maximum_norm(stdQuGate->getScalarGM()->matrix(),
                                         gate->getScalarGM()->matrix()),
                      0.0,
                      title,
                      info);
  };

  gate0 = StandardQuantumGate::Create(ScalarGateMatrix::I1(), nullptr, {0});
  gate1 = StandardQuantumGate::Create(ScalarGateMatrix::I1(), nullptr, {1});
  gate = StandardQuantumGate::Create(ScalarGateMatrix::I2(), nullptr, {0, 1});
  check("I1 * I1 = I2", GET_INFO());

  gate0 = StandardQuantumGate::Create(ScalarGateMatrix::I1(), nullptr, {0});
  gate1 = StandardQuantumGate::Create(ScalarGateMatrix::H(), nullptr, {1});
  // clang-format off
  ComplexSquareMatrix matHI{
    // real
    { 
      M_SQRT1_2, 0.0, M_SQRT1_2, 0.0,
      0.0, M_SQRT1_2, 0.0, M_SQRT1_2,
      M_SQRT1_2, 0.0, -M_SQRT1_2, 0.0,
      0.0, 0.0, M_SQRT1_2, 0.0, -M_SQRT1_2
    },
    // imag
    {
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0
    }
  };
  ComplexSquareMatrix matIH{
    // real
    {
      M_SQRT1_2, M_SQRT1_2, 0.0, 0.0,
      M_SQRT1_2,-M_SQRT1_2, 0.0, 0.0,
      0.0, 0.0, M_SQRT1_2, M_SQRT1_2,
      0.0, 0.0, M_SQRT1_2,-M_SQRT1_2
    },
    // imag
    {
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0
    }
  };
  // clang-format on
  gate = StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(matHI), nullptr, {0, 1});
  check("H(1) otimes I1(1) = H(1)I1(0)", GET_INFO());

  gate0 = StandardQuantumGate::Create(ScalarGateMatrix::I1(), nullptr, {2});
  gate1 = StandardQuantumGate::Create(ScalarGateMatrix::H(), nullptr, {3});
  check("H(3) otimes I1(2) = H(3)I1(2)", GET_INFO());

  gate0 = StandardQuantumGate::Create(ScalarGateMatrix::H(), nullptr, {0});
  gate1 = StandardQuantumGate::Create(ScalarGateMatrix::I1(), nullptr, {1});

  gate = StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(matIH), nullptr, {0, 1});

  check("I1 otimes H = IH", GET_INFO());

  // clang-format off
  ComplexSquareMatrix mat0{
    // real
    {1.0, 2.0, 3.0, 4.0},
    // imag
    {0.0, 0.0, 0.0, 0.0}
  };
  ComplexSquareMatrix mat00{
    // real
    {7.0, 10.0, 15.0, 22.0},
    // imag
    {0.0, 0.0, 0.0, 0.0}
  };
  ComplexSquareMatrix mat0_otimes_0{
    // real
    {
      1, 2, 2, 4,
      3, 4, 6, 8,
      3, 6, 4, 8,
      9, 12, 12, 16
    },
    // imag
    {
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0
    }
  };
  // clang-format on
  gate0 = StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(mat0), nullptr, {0});
  gate1 = gate0;
  gate = StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(mat00), nullptr, {0});
  check("Arbitrary matrix example", GET_INFO());
  gate1 = StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(mat0), nullptr, {1});
  gate = StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(mat0_otimes_0), nullptr, {0, 1});
  check("Arbitrary matrix example with 2 qubits", GET_INFO());

  ComplexSquareMatrix mat1{// real
                           {0.0, 0.0, 0.0, 0.0},
                           // imag
                           {1.0, 2.0, 3.0, 4.0}};
  ComplexSquareMatrix mat11{// real
                            {-7.0, -10.0, -15.0, -22.0},
                            // imag
                            {0.0, 0.0, 0.0, 0.0}};
  ComplexSquareMatrix mat1_otimes_1 = mat0_otimes_0 * -1.0;
  gate0 = StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(mat1), nullptr, {0});
  gate1 = gate0;
  gate = StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(mat11), nullptr, {0});
  check("Arbitrary matrix example 2", GET_INFO());

  gate1 = StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(mat1), nullptr, {1});
  gate = StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(mat1_otimes_1), nullptr, {0, 1});
  check("Arbitrary matrix example 2 with 2 qubits", GET_INFO());

  ComplexSquareMatrix mat2{// real
                           {1.0, 2.0, 3.0, 4.0},
                           // imag
                           {5.0, 6.0, 7.0, 8.0}};

  ComplexSquareMatrix mat22{// real
                            {-60, -68, -76, -84},
                            // imag
                            {42, 56, 74, 96}};

  ComplexSquareMatrix mat2_otimes_2{
      // real
      {-24,
       -28,
       -28,
       -32,
       -32,
       -36,
       -36,
       -40,
       -32,
       -36,
       -36,
       -40,
       -40,
       -44,
       -44,
       -48},
      // imag
      {10, 16, 16, 24, 22, 28, 32, 40, 22, 32, 28, 40, 42, 52, 52, 64}};

  gate0 = StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(mat2), nullptr, {0});
  gate1 = gate0;
  gate = StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(mat22), nullptr, {0});
  check("Arbitrary matrix example 3", GET_INFO());
  gate1 = StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(mat2), nullptr, {1});
  gate = StandardQuantumGate::Create(
      std::make_shared<ScalarGateMatrix>(mat2_otimes_2), nullptr, {0, 1});
  check("Arbitrary matrix example 3 with 2 qubits", GET_INFO());

  suite.displayResult();
}
