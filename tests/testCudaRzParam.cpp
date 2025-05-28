#include "simulation/KernelManager.h"
#include "tests/TestKit.h"
#include "simulation/StatevectorCPU.h"
#include "simulation/StatevectorCUDA.h"
#include <random>
#include <cmath>

using namespace cast;
using namespace cast::test;

namespace {
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
}

template<typename SV>
static void preparePlusState(SV &sv, int q)
{
    sv.initialize();
    double norm = 1.0 / std::sqrt(2.0);

    // host-side edit of the two amplitudes that differ
    auto *h = sv.hData(); // host pointer
    h[0] = norm; // |…0>
    h[1ull << q] = norm; // |…1>

    // push to device if this is a CUDA SV
    if constexpr (std::is_same_v<SV, utils::StatevectorCUDA<double>>) {
        std::size_t bytes = (1ULL << sv.nQubits()) * 2 * sizeof(double);
        CUDA_CALL(cudaMemcpy(sv.dData(), h, bytes, cudaMemcpyHostToDevice),
                  "cpy |+> to device");
    }
}

static std::complex<double> hostAmp(const utils::StatevectorCUDA<double>& sv,
                                    std::size_t idx)
{
  const double* h = sv.hData();
  return { h[2*idx], h[2*idx+1] };
}

static double relativePhase(utils::StatevectorCUDA<double>& sv, int q)
{
  std::size_t bytes = (1ULL << sv.nQubits()) * 2 * sizeof(double);
  CUDA_CALL(cudaMemcpy(const_cast<double*>(sv.hData()), sv.dData(), bytes, cudaMemcpyDeviceToHost), "cpy device→host");
  std::complex<double> a0 = hostAmp(sv, 0);
  std::complex<double> a1 = hostAmp(sv, 1ull << q);
  return std::arg(a1) - std::arg(a0);
}

template<unsigned nQubits>
static void f() {
  test::TestSuite suite(
    "Symbolic Rz param gate (CUDA) (" + std::to_string(nQubits) + " qubits)");

  CUDAKernelManager kernelMgrCUDA;

  const int nGates = 3;
  std::vector<std::shared_ptr<QuantumGate>> gates(nGates);
  for (int i = 0; i < nGates; ++i) {
    gates[i] = getRzSymbolicGate(i);  // Rz on qubit i
  }

  // 1: default memory space
  {
    CUDAKernelGenConfig cudaConfig;
    cudaConfig.precision = 64; // double
    cudaConfig.matrixLoadMode = CUDAKernelGenConfig::LoadInDefaultMemSpace;

    for (int i = 0; i < nGates; i++) {
      kernelMgrCUDA.genCUDAGate(
          cudaConfig, gates[i],
          "rz_param_gate_def_" + std::to_string(i));
    }
  }

  // 2: constant memory space
  {
    CUDAKernelGenConfig cudaConfig;
    cudaConfig.precision = 64; // double
    cudaConfig.matrixLoadMode = CUDAKernelGenConfig::LoadInConstMemSpace;

    for (int i = 0; i < nGates; i++) {
      kernelMgrCUDA.genCUDAGate(
          cudaConfig, gates[i],
          "rz_param_gate_const_" + std::to_string(i));
    }
  }

  // Now we have 2 * nGates = 6 kernels total, can compile them all at once
  kernelMgrCUDA.emitPTX(/*nThreads=*/2, llvm::OptimizationLevel::O1, /*verbose=*/0);
  // llvm::outs() << "\n=== DUMPING PTX FOR INSPECTION ===\n";
  // kernelMgrCUDA.dumpPTX("rz_param_gate_def_0", llvm::outs());
  // kernelMgrCUDA.dumpPTX("rz_param_gate_const_0", llvm::outs());
  // llvm::outs() << "=== END PTX DUMP ===\n\n";
  kernelMgrCUDA.initCUJIT(/*nThreads=*/2, /*verbose=*/0);

  utils::StatevectorCUDA<double> sv(nQubits);

  // Build a numeric Rz(π/2) matrix, allocate & copy to device
  const double theta = M_PI / 2.0;
  auto numericMat = buildRzNumericMatrix(theta);

  double* dMatPtr = nullptr;
  CUDA_CALL(cudaMalloc(&dMatPtr, numericMat.size()*sizeof(double)),
            "cudaMalloc RzMatrix");
  CUDA_CALL(cudaMemcpy(dMatPtr, numericMat.data(),
                       numericMat.size()*sizeof(double),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy RzMatrix");


  // Indexes [0..2] are defaultMem, [3..5] are constMem

  // Test the 3 defaultMem kernels
  for (int i = 0; i < nGates; i++) {
    preparePlusState(sv, i);
    // launch
    kernelMgrCUDA.launchCUDAKernelParam(
      sv.dData(),
      sv.nQubits(),
      kernelMgrCUDA.kernels()[i],
      dMatPtr,
      64
    );
    CUDA_CALL(cudaDeviceSynchronize(), "sync after kernel");
    double phase = relativePhase(sv, i);
    if (phase >  M_PI) phase -= 2*M_PI;
    if (phase < -M_PI) phase += 2*M_PI;

    suite.assertClose(phase, -theta/2,
      "phase check (DefaultMem)", GET_INFO(), 1e-11);
    suite.assertClose(sv.norm(), 1.0,
      "Norm (DefaultMem)", GET_INFO(), 1e-12);
  }

  // Test the 3 constMem kernels
  for (int i = 0; i < nGates; i++) {
    int kernelIndex = i + nGates;

    preparePlusState(sv, i);

    kernelMgrCUDA.launchCUDAKernelParam(
      sv.dData(),
      sv.nQubits(),
      kernelMgrCUDA.kernels()[kernelIndex],
      dMatPtr,
      64
    );
    CUDA_CALL(cudaDeviceSynchronize(), "sync after kernel");

    double phase = relativePhase(sv, i);
    if (phase >  M_PI) phase -= 2*M_PI;
    if (phase < -M_PI) phase += 2*M_PI;

    suite.assertClose(phase, -theta/2,
        "phase check (ConstMem)", GET_INFO(), 1e-11);
    suite.assertClose(sv.norm(), 1.0,
        "Norm (ConstMem)", GET_INFO(), 1e-12);
  }

  // 8) Cleanup
  CUDA_CALL(cudaFree(dMatPtr), "cudaFree RzMatrix");
  suite.displayResult();
}

void test::test_cudaRz_param() {
  f<8>();
  f<12>();
}
