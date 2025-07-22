// #include "simulation/KernelManager.h"
// #include "tests/TestKit.h"
// #include "simulation/StatevectorCPU.h"
// #include "simulation/StatevectorCUDA.h"
// #include <random>

// using namespace cast;
// using namespace cast::test;

// namespace {
//   GateMatrix makeRzSymbolicMatrix() {
//     GateMatrix::p_matrix_t pMat(2);

//     int var = 0; 
//     Polynomial c( Monomial::Cosine(var) );
//     Polynomial s( Monomial::Sine(var) );

//     Polynomial minusI( std::complex<double>(0.0, -1.0) );
//     Polynomial minusI_s = minusI * s;

//     // Rz(θ) = [[ c, -i s ],
//     //          [ -i s, c ]]
//     pMat(0,0) = c;
//     pMat(0,1) = minusI_s;
//     pMat(1,0) = minusI_s;
//     pMat(1,1) = c;

//     // Wrapping in a GateMatrix => isConvertibleToCMat = UnConvertible
//     // => getConstantMatrix() = null
//     GateMatrix gmat(pMat);
//     return gmat;
//   }

//   std::shared_ptr<QuantumGate> getRzSymbolicGate(int q) {
//     GateMatrix rzSymbolic = makeRzSymbolicMatrix();
//     QuantumGate gate(rzSymbolic, q);
//     return std::make_shared<QuantumGate>(gate);
//   }

//   std::vector<double> buildRzNumericMatrix(double theta) {
//     double c = std::cos(theta / 2.0);
//     double s = std::sin(theta / 2.0);

//     std::vector<double> mat(8);
//     mat[0] = c;    // re(0,0)
//     mat[1] = 0.0;  // im(0,0)
//     mat[2] = 0.0;  // re(0,1)
//     mat[3] = -s;   // im(0,1)
//     mat[4] = 0.0;  // re(1,0)
//     mat[5] = -s;   // im(1,0)
//     mat[6] = c;    // re(1,1)
//     mat[7] = 0.0;  // im(1,1)
//     return mat;
//   }
// }

// template<unsigned nQubits>
// static void f() {
//   test::TestSuite suite(
//     "Symbolic Rz param gate (CUDA) (" + std::to_string(nQubits) + " qubits)");

//   CUDAKernelManager kernelMgrCUDA;

//   const int nGates = 3;
//   std::vector<std::shared_ptr<QuantumGate>> gates(nGates);
//   for (int i = 0; i < nGates; ++i) {
//     gates[i] = getRzSymbolicGate(i);  // Rz on qubit i
//   }

//   // 1: default memory space
//   {
//     CUDAKernelGenConfig cudaConfig;
//     cudaConfig.precision = 64; // double
//     cudaConfig.matrixLoadMode = CUDAKernelGenConfig::LoadInDefaultMemSpace;

//     for (int i = 0; i < nGates; i++) {
//       kernelMgrCUDA.genCUDAGate(
//           cudaConfig, gates[i],
//           "rz_param_gate_def_" + std::to_string(i), nQubits);
//     }
//   }

//   // 2: constant memory space
//   {
//     CUDAKernelGenConfig cudaConfig;
//     cudaConfig.precision = 64; // double
//     cudaConfig.matrixLoadMode = CUDAKernelGenConfig::LoadInConstMemSpace;

//     for (int i = 0; i < nGates; i++) {
//       kernelMgrCUDA.genCUDAGate(
//           cudaConfig, gates[i],
//           "rz_param_gate_const_" + std::to_string(i), nQubits);
//     }
//   }

//   // Now we have 2 * nGates = 6 kernels total, can compile them all at once
//   kernelMgrCUDA.emitPTX(/*nThreads=*/2, llvm::OptimizationLevel::O1, /*verbose=*/0);
//   // llvm::outs() << "\n=== DUMPING PTX FOR INSPECTION ===\n";
//   // kernelMgrCUDA.dumpPTX("rz_param_gate_def_0", llvm::outs());
//   // kernelMgrCUDA.dumpPTX("rz_param_gate_const_0", llvm::outs());
//   // llvm::outs() << "=== END PTX DUMP ===\n\n";
//   kernelMgrCUDA.initCUJIT(/*nThreads=*/2, /*verbose=*/0);

//   utils::StatevectorCUDA<double> sv(nQubits);

//   // Build a numeric Rz(π/2) matrix, allocate & copy to device
//   double theta = M_PI / 2.0;
//   auto numericMat = buildRzNumericMatrix(theta);

//   double* dMatPtr = nullptr;
//   CUDA_CALL(cudaMalloc(&dMatPtr, numericMat.size()*sizeof(double)),
//             "cudaMalloc RzMatrix");
//   CUDA_CALL(cudaMemcpy(dMatPtr, numericMat.data(),
//                        numericMat.size()*sizeof(double),
//                        cudaMemcpyHostToDevice),
//             "cudaMemcpy RzMatrix");


//   // Indexes [0..2] are defaultMem, [3..5] are constMem

//   // Test the 3 defaultMem kernels
//   for (int i = 0; i < nGates; i++) {
//     // Initialize
//     sv.initialize();
//     suite.assertClose(sv.norm(), 1.0, "Init Norm (DefaultMem)", GET_INFO());
//     suite.assertClose(sv.prob(i), 0.0, "Init Prob("+std::to_string(i)+")", GET_INFO());

//     // launch
//     kernelMgrCUDA.launchCUDAKernelParam(
//       sv.dData(),
//       sv.nQubits(),
//       kernelMgrCUDA.kernels()[i],
//       dMatPtr,
//       64
//     );

//     suite.assertClose(sv.norm(), 1.0,
//         "After Rz("+std::to_string(i)+") param: Norm (DefaultMem)", GET_INFO());
//   }

//   // Test the 3 constMem kernels
//   for (int i = 0; i < nGates; i++) {
//     // The constMem kernels are at index i + nGates = i + 3
//     int kernelIndex = i + nGates;

//     // re-initialize
//     sv.initialize();
//     suite.assertClose(sv.norm(), 1.0, "Init Norm (ConstMem)", GET_INFO());
//     suite.assertClose(sv.prob(i), 0.0, "Init Prob("+std::to_string(i)+")", GET_INFO());

//     kernelMgrCUDA.launchCUDAKernelParam(
//       sv.dData(),
//       sv.nQubits(),
//       kernelMgrCUDA.kernels()[kernelIndex],
//       dMatPtr,
//       64
//     );

//     suite.assertClose(sv.norm(), 1.0,
//         "After Rz("+std::to_string(i)+") param: Norm (ConstMem)", GET_INFO());
//   }

//   // 8) Cleanup
//   CUDA_CALL(cudaFree(dMatPtr), "cudaFree RzMatrix");
//   suite.displayResult();
// }

// void test::test_cudaRz_param() {
//   f<8>();
//   f<12>();
// }



#include "simulation/KernelManager.h"
#include "tests/TestKit.h"
#include "simulation/StatevectorCPU.h"
#include "simulation/StatevectorCUDA.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

using namespace cast;
using namespace cast::test;

namespace {

std::vector<double> buildRzNumericVector(double theta) {
  double c = std::cos(theta / 2.0);
  double s = std::sin(theta / 2.0);
  return { c, -s, 0.0, 0.0, 0.0, 0.0, c, s };
}

/* build a *constant* GateMatrix from θ for the CPU reference. */
GateMatrix makeRzConstantGateMatrix(double theta)
{
  GateMatrix::c_matrix_t cMat(2);

  const double c = std::cos(theta / 2.0);
  const double s = std::sin(theta / 2.0);

  cMat(0,0) = std::complex<double>(c, -s);
  cMat(0,1) = std::complex<double>(0.0, 0.0);
  cMat(1,0) = std::complex<double>(0.0, 0.0);
  cMat(1,1) = std::complex<double>(c, s);

  return GateMatrix(cMat); // convertible: getConstantMatrix() works
}

/* Symbolic Rz(θ) gate (sine/cosine polynomials) */
GateMatrix makeRzSymbolicMatrix()
{
  GateMatrix::p_matrix_t pMat(2);

  int var = 0;
  Polynomial c (Monomial::Cosine(var));
  Polynomial s (Monomial::Sine  (var));
  Polynomial minusI  (std::complex<double>(0.0, -1.0));
  Polynomial minusI_s = minusI * s;

  // Rz(θ) = [[ c, -i s ],
  //          [ -i s, c ]]
  pMat(0,0) = c;
  pMat(0,1) = minusI_s;
  pMat(1,0) = minusI_s;
  pMat(1,1) = c;

  return GateMatrix(pMat); // NOT convertible: symbolic kernel
}

std::shared_ptr<QuantumGate> makeSymbolicRzGate(int q)
{
  return std::make_shared<QuantumGate>(makeRzSymbolicMatrix(), q);
}

#ifdef VERBOSE
void dumpStatevector(const char* tag,
                     const std::vector<std::complex<double>>& v)
{
  std::cerr << tag << '\n';
  for (std::size_t i = 0; i < v.size(); ++i)
    std::cerr << "  │" << std::setw(3) << i << "⟩ = ("
              << v[i].real() << ',' << v[i].imag() << ")\n";
}
#else
inline void dumpStatevector(const char*, const std::vector<std::complex<double>>&){}
#endif

} // namespace

template<unsigned nQubits>
static void runRzParamTest()
{
  test::TestSuite suite("Symbolic-parameter Rz test (" + std::to_string(nQubits) + " qubits)");

  // Limit nGates to the number of available qubits
  const int nGates = std::min(3, static_cast<int>(nQubits));
  std::vector<std::shared_ptr<QuantumGate>> gates;
  for (int q = 0; q < nGates; ++q) {
    gates.push_back(makeSymbolicRzGate(q));  // Rz on qubit q
  }

  // ── Generate two kernel variants per gate (default-mem / const-mem)
  CUDAKernelManager kmgr;
  for (int mode = 0; mode < 2; ++mode) {
    CUDAKernelGenConfig cudaConfig;
    cudaConfig.precision = 64; // double
    cudaConfig.matrixLoadMode = (mode == 0)
        ? CUDAKernelGenConfig::LoadInDefaultMemSpace
        : CUDAKernelGenConfig::LoadInConstMemSpace;
    // cudaConfig.matrixLoadMode = CUDAKernelGenConfig::LoadInConstMemSpace;

    for (int i = 0; i < nGates; ++i) {
      kmgr.genCUDAGate(
          cudaConfig, gates[i],
          (mode == 0 ? "rz_def_" : "rz_const_") + std::to_string(i), nQubits);
    }
  }

  // Now we have 2 * nGates kernels total, can compile them all at once
  kmgr.emitPTX(/*nThreads=*/2, llvm::OptimizationLevel::O1, /*verbose=*/0);
  // llvm::outs() << "\n=== DUMPING PTX FOR INSPECTION ===\n";
  // kernelMgrCUDA.dumpPTX("rz_param_gate_def_0", llvm::outs());
  // kernelMgrCUDA.dumpPTX("rz_param_gate_const_0", llvm::outs());
  // llvm::outs() << "=== END PTX DUMP ===\n\n";
  kmgr.initCUJIT(/*nThreads=*/2, /*verbose=*/0);

  // ── GPU / CPU state-vectors
  utils::StatevectorCUDA<double> svCUDA(nQubits);
  utils::StatevectorCPU<double>  svCPU(nQubits, /*simd_s*/0);

  // ── Device buffer for the numeric matrix
  double* dMat = nullptr;
  CUDA_CALL(cudaMalloc(&dMat, 8*sizeof(double)), "cudaMalloc Rz matrix");

  // Build a numeric Rz(π/2) matrix, allocate & copy to device
  // const std::vector<double> thetas = {0.0, M_PI/3.0, M_PI/2.0, -M_PI/2.0};
  const std::vector<double> thetas = {M_PI/2.0};

  for (double theta : thetas)
  {
    /* Upload numeric Rz(θ) to device */
    auto hostVec = buildRzNumericVector(theta);
    CUDA_CALL(cudaMemcpy(dMat, hostVec.data(), 8*sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy Rz matrix");

    // Debugging: Verify the numeric matrix on device
    std::vector<double> debugMat(8);
    CUDA_CALL(cudaMemcpy(debugMat.data(), dMat, 8 * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy debug Rz matrix");
    // std::cout << "Debug: Numeric Rz Matrix (theta=" << theta << ") on Device:\n";
    // for (size_t i = 0; i < 8; i += 2) {
    //   std::cout << "Element (" << (i / 2) / 2 << "," << (i / 2) % 2 << "): "
    //             << debugMat[i] << " + " << debugMat[i + 1] << "i\n";
    // }

    // prepare constant GateMatrix for CPU reference
    GateMatrix rzCPUmat = makeRzConstantGateMatrix(theta);

    for (int gateIdx = 0; gateIdx < nGates; ++gateIdx) {
      const int targetQ = gateIdx;

      // Indexes [0..(nGates-1)] are defaultMem, [nGates..(2*nGates-1)] are constMem
      for (int variant = 0; variant < 2; ++variant) {
        // The constMem kernels are at index i + nGates
        const int kIndex = gateIdx + variant * nGates;

        // Randomise GPU SV and copy to CPU mirror
        svCUDA.randomize();
        CUDA_CALL(cudaMemcpy(svCPU.data(), svCUDA.dData(), svCUDA.sizeInBytes(), cudaMemcpyDeviceToHost), "cudaMemcpy rand SV");

        suite.assertClose(svCUDA.norm(), 1.0,
                          "Initial norm = 1", GET_INFO());

        // Debugging: Dump initial state vector
        if (nQubits <= 4) {
          std::vector<std::complex<double>> initialSV(1ULL << nQubits);
          CUDA_CALL(cudaMemcpy(initialSV.data(), svCUDA.dData(), svCUDA.sizeInBytes(), cudaMemcpyDeviceToHost), "cudaMemcpy initial SV");
          // std::cout << "Debug: Initial GPU State Vector (targetQ=" << targetQ << ", variant=" << variant << "):\n";
          // for (size_t i = 0; i < initialSV.size(); ++i) {
          //   std::cout << "Index " << i << ": " << initialSV[i] << "\n";
          // }
        }

        // Debugging: Log kernel launch parameters
        // std::cout << "Debug: Launching kernel " << kIndex << " (targetQ=" << targetQ << ", variant=" << variant << ")\n";
        // std::cout << "nQubits=" << nQubits << ", blockSize=64\n";

        // ── GPU: launch symbolic-parameter kernel
        kmgr.launchCUDAKernelParam(
            svCUDA.dData(), svCUDA.nQubits(),
            kmgr.kernels()[kIndex],
            dMat, /*sharedMem*/64);
        cudaDeviceSynchronize();

        suite.assertClose(svCUDA.norm(), 1.0, "Post-GPU norm = 1", GET_INFO());

        // Debugging: Dump GPU state vector after kernel
        if (nQubits <= 4) {
          std::vector<std::complex<double>> gpuSV(1ULL << nQubits);
          CUDA_CALL(cudaMemcpy(gpuSV.data(), svCUDA.dData(), svCUDA.sizeInBytes(), cudaMemcpyDeviceToHost), "cudaMemcpy GPU SV");
          // std::cout << "Debug: GPU State Vector after Rz (targetQ=" << targetQ << ", variant=" << variant << "):\n";
          // for (size_t i = 0; i < gpuSV.size(); ++i) {
          //   std::cout << "Index " << i << ": " << gpuSV[i] << "\n";
          // }
        }

        // CPU: apply numeric Rz(θ)
        QuantumGate cpuGate(rzCPUmat, targetQ);
        svCPU.applyGate(cpuGate);

        // Debugging: Dump CPU state vector
        // if (nQubits <= 4) {
        //   std::cout << "Debug: CPU State Vector after Rz (targetQ=" << targetQ << ", variant=" << variant << "):\n";
        //   for (size_t i = 0; i < (1ULL << nQubits); ++i) {
        //     std::complex<double> cpuAmp(svCPU.data()[2 * i], svCPU.data()[2 * i + 1]);
        //     std::cout << "Index " << i << ": " << cpuAmp << "\n";
        //   }
        // }

        // Compare per-qubit probabilities
        for (int q = 0; q < static_cast<int>(nQubits); ++q) {
          suite.assertClose(svCUDA.prob(q), svCPU.prob(q),
              "Prob(q=" + std::to_string(q) + ") match (θ="
              + std::to_string(theta) + ", var="
              + std::to_string(variant) + ')',
              GET_INFO());
        }

        // Global fidelity
        {
          std::vector<std::complex<double>> hostCUDA(1ULL << nQubits);
          CUDA_CALL(cudaMemcpy(hostCUDA.data(), svCUDA.dData(), svCUDA.sizeInBytes(), cudaMemcpyDeviceToHost), "cudaMemcpy final SV");

          std::complex<double> inner = 0;
          for (std::size_t i = 0; i < hostCUDA.size(); ++i) {
            std::complex<double> cpuAmp(svCPU.data()[2*i],
                                        svCPU.data()[2*i+1]);
            inner += std::conj(cpuAmp) * hostCUDA[i];
          }
          suite.assertClose(std::abs(inner), 1.0,
              "Fidelity ≈ 1 (θ=" + std::to_string(theta) +
              ", var=" + std::to_string(variant) + ')',
              GET_INFO());
        }
      }
    }
  }

  // Cleanup
  CUDA_CALL(cudaFree(dMat), "cudaFree Rz matrix");
  suite.displayResult();
}

void test::test_cudaRz_param()
{
  runRzParamTest<1>();
  runRzParamTest<2>();
  runRzParamTest<4>();
  runRzParamTest<8>();
}