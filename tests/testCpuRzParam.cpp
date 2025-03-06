#include "simulation/KernelManager.h"
#include "tests/TestKit.h"
#include "utils/statevector.h"

#include <cmath>
#include <vector>
#include <memory>
#include <string>

using namespace cast;
using namespace utils;


GateMatrix makeRzSymbolicMatrix() {
    GateMatrix::p_matrix_t pMat(2);

    int var = 0; 
    Polynomial c( Monomial::Cosine(var) );
    Polynomial s( Monomial::Sine(var) );

    Polynomial minusI( std::complex<double>(0.0, -1.0) );
    Polynomial minusI_s = minusI * s;

    // Rz(θ) = [[ c, -i s ],
    //          [ -i s, c ]]
    pMat(0,0) = c;
    pMat(0,1) = minusI_s;
    pMat(1,0) = minusI_s;
    pMat(1,1) = c;

    // Wrapping in a GateMatrix => isConvertibleToCMat = UnConvertible
    // => getConstantMatrix() = null
    GateMatrix gmat(pMat);
    return gmat;
}

std::shared_ptr<QuantumGate> getRzSymbolicGate(int q) {
    GateMatrix rzSymbolic = makeRzSymbolicMatrix();
    QuantumGate gate(rzSymbolic, q);
    return std::make_shared<QuantumGate>(gate);
}

std::vector<double> buildRzNumericMatrix(double theta) {
    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);

    // Rz(θ) = [[ c,   -i s ],
    //          [ -i s,   c ]]
    // => (0,0) = c + i0
    // => (0,1) = 0 + i(-s)
    // => (1,0) = 0 + i(-s)
    // => (1,1) = c + i0
    std::vector<double> mat(8);
    mat[0] = c;    // re(0,0)
    mat[1] = 0.0;  // im(0,0)
    mat[2] = 0.0;  // re(0,1)
    mat[3] = -s;   // im(0,1)
    mat[4] = 0.0;  // re(1,0)
    mat[5] = -s;   // im(1,0)
    mat[6] = c;    // re(1,1)
    mat[7] = 0.0;  // im(1,1)
    return mat;
}

template<unsigned simd_s>
static void f() {
    test::TestSuite suite("Symbolic Rz param gates (s = " + std::to_string(simd_s) + ")");

    CPUKernelManager cpuKernelMgr;
    CPUKernelGenConfig cpuConfig;
    cpuConfig.simd_s = simd_s;
    cpuConfig.matrixLoadMode = CPUKernelGenConfig::StackLoadMatElems;

    cpuKernelMgr.genCPUGate(cpuConfig, getRzSymbolicGate(0), "gate_rz_0_param");
    cpuKernelMgr.genCPUGate(cpuConfig, getRzSymbolicGate(1), "gate_rz_1_param");
    cpuKernelMgr.genCPUGate(cpuConfig, getRzSymbolicGate(2), "gate_rz_2_param");
    cpuKernelMgr.genCPUGate(cpuConfig, getRzSymbolicGate(3), "gate_rz_3_param");

    // cpuKernelMgr.dumpIR("gate_rz_0_param", llvm::outs());

    cpuKernelMgr.initJIT();

    utils::StatevectorAlt<double> sv(6, simd_s);

    double theta = M_PI / 2.0;
    auto numericMat = buildRzNumericMatrix(theta);

    // Rz on qubit 0
    sv.initialize();
    suite.assertClose(sv.norm(), 1.0, "Init Norm [q=0]", GET_INFO());
    suite.assertClose(sv.prob(0), 0.0, "Init Prob(0)", GET_INFO());

    cpuKernelMgr.applyCPUKernel(
        sv.data,
        sv.nQubits,
        "gate_rz_0_param",
        numericMat.data()
    );
    suite.assertClose(sv.norm(), 1.0, "After Rz(0) param: Norm", GET_INFO());

    // Rz on qubit 1
    sv.initialize();
    suite.assertClose(sv.norm(), 1.0, "Init Norm [q=1]", GET_INFO());
    suite.assertClose(sv.prob(1), 0.0, "Init Prob(1)", GET_INFO());

    cpuKernelMgr.applyCPUKernel(
        sv.data,
        sv.nQubits,
        "gate_rz_1_param",
        numericMat.data()
    );
    suite.assertClose(sv.norm(), 1.0, "After Rz(1) param: Norm", GET_INFO());

    // Rz on qubit 2
    sv.initialize();
    suite.assertClose(sv.norm(), 1.0, "Init Norm [q=2]", GET_INFO());
    suite.assertClose(sv.prob(2), 0.0, "Init Prob(2)", GET_INFO());
    cpuKernelMgr.applyCPUKernel(
        sv.data,
        sv.nQubits,
        "gate_rz_2_param",
        numericMat.data()
    );
    suite.assertClose(sv.norm(), 1.0, "After Rz(2) param: Norm", GET_INFO());

    // Rz on qubit 3
    sv.initialize();
    suite.assertClose(sv.norm(), 1.0, "Init Norm [q=3]", GET_INFO());
    suite.assertClose(sv.prob(3), 0.0, "Init Prob(3)", GET_INFO());
    cpuKernelMgr.applyCPUKernel(
        sv.data,
        sv.nQubits,
        "gate_rz_3_param",
        numericMat.data()
    );
    suite.assertClose(sv.norm(), 1.0, "After Rz(3) param: Norm", GET_INFO());

    suite.displayResult();
}

void test::test_cpuRz_param() {
    f<1>();
    f<2>();
}
