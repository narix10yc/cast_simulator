#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "timeit/timeit.h"
#include "llvm/Passes/OptimizationLevel.h"

#include "cast/Core/IRNode.h"
#include "cast/Core/KernelGenInternal.h"

#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "cast/CUDA/Config.h"
#include "cast/CUDA/CUDAPermute.h"
#include "cast/CUDA/CUDAOptimizer.h"
#include "cast/CUDA/CUDAFusionConfig.h"
#include "cast/CUDA/CUDACostModel.h"

using namespace cast;

static void printTimingStats(const std::string& label,
                             const timeit::TimingResult& tr) {
  using std::cout;
  if (tr.tArr.empty()) { cout << label << ": No data\n"; return; }
  double sum = 0.0; for (double t : tr.tArr) sum += t;
  double mean = sum / tr.tArr.size();
  double sumsq = 0.0; for (double t : tr.tArr) { double d=t-mean; sumsq += d*d; }
  double stdev = std::sqrt(tr.tArr.size()>1 ? sumsq/tr.tArr.size() : 0.0);
  std::vector<double> s = tr.tArr; std::sort(s.begin(), s.end());
  double med = s.size()%2 ? s[s.size()/2] : 0.5*(s[s.size()/2-1]+s[s.size()/2]);

  cout << std::fixed << std::setprecision(3)
       << label << ": mean=" << (mean*1e3) << " ms ± " << (stdev*1e3)
       << " ms (median=" << (med*1e3) << " ms, n=" << s.size() << ")\n";
}

struct Args {
  // std::string qasmFile = "../examples/qft/qft-16-cp.qasm";
  // std::string qasmFile = "../examples/rqc/q12_189_128.qasm";
  // std::string qasmFile = "../examples/rqc/q20_592_427.qasm";
  // std::string qasmFile  = "../examples/rqc/q28_442_300.qasm";
  std::string qasmFile = "../examples/rqc/q30_521_379.qasm";
  // std::string qasmFile  = "../examples/rqc/q30_4299_3272.qasm";
  std::string graphName = "testCircuit";
  int reps = 3;
  int blockSize = 256;
  int nThreads = 1;             // worker threads for JIT/emit
  int precBits = 64;            // 32 or 64
  bool fuse = true;             // fusion ON by default
  std::string perfCSV;          // path to *GPU* performance CSV for fusion
  llvm::OptimizationLevel opt = llvm::OptimizationLevel::O1;
};

static Args parseCLI(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string v(argv[i]);
    auto next = [&](int& i)->std::string{
      if (i+1 >= argc) { std::cerr<<"Missing value after "<<v<<"\n"; std::exit(1); }
      return std::string(argv[++i]);
    };
    if (v=="-i"||v=="--input") a.qasmFile = next(i);
    else if (v=="--reps")      a.reps = std::stoi(next(i));
    else if (v=="--block")     a.blockSize = std::stoi(next(i));
    else if (v=="--threads")   a.nThreads = std::stoi(next(i));
    else if (v=="--prec")      a.precBits = std::stoi(next(i));
    else if (v=="--no-fusion") a.fuse = false;
    else if (v=="--perf-csv")  a.perfCSV = next(i);
    else if (v=="--opt") {
      std::string s = next(i);
      if      (s=="O0") a.opt=llvm::OptimizationLevel::O0;
      else if (s=="O1") a.opt=llvm::OptimizationLevel::O1;
      else if (s=="O2") a.opt=llvm::OptimizationLevel::O2;
      else if (s=="O3") a.opt=llvm::OptimizationLevel::O3;
      else { std::cerr<<"Unknown --opt "<<s<<"\n"; std::exit(1); }
    } else {
      std::cerr<<"Unknown arg: "<<v<<"\n"; std::exit(1);
    }
  }
  return a;
}

/* ================= PERMUTE HELPERS (LSB placement) ================== */
#ifdef CAST_USE_CUDA
#include <utility>

static inline void set_layout_LSB(cast::BitLayout& layout,
                                  const std::vector<int>& Q,
                                  int nSys) {
  const int k = (int)Q.size();
  std::vector<char> isTarget(nSys, 0);
  for (int b = 0; b < k; ++b) isTarget[Q[b]] = 1;

  // non-targets in ascending old physical order
  std::vector<std::pair<int,int>> others;
  others.reserve(std::max(0, nSys - k));
  for (int l = 0; l < nSys; ++l)
    if (!isTarget[l]) others.emplace_back(layout.phys_of_log[l], l);
  std::sort(others.begin(), others.end());

  for (int b = 0; b < k; ++b) layout.phys_of_log[Q[b]] = b;
  for (int i = 0; i < (int)others.size(); ++i)
    layout.phys_of_log[others[i].second] = k + i;

  for (int p = 0; p < nSys; ++p) layout.log_of_phys[p] = -1;
  for (int l = 0; l < nSys; ++l) layout.log_of_phys[layout.phys_of_log[l]] = l;
}

template<typename ScalarType>
static inline void permute_to_LSBs_if_needed(
    ScalarType*& dCurrent,
    ScalarType*  dScratch,
    int          nSys,
    const std::vector<int>& logicalQubits,
    cast::BitLayout& layout,
    cudaStream_t stream = 0)
{
  const int k = (int)logicalQubits.size();
  if (k == 0) return;

  bool lsbOK = true;
  for (int b = 0; b < k; ++b)
    if (layout.phys_of_log[logicalQubits[b]] != b) { lsbOK = false; break; }
  if (lsbOK) return;

  uint64_t maskLow = 0;
  for (int b = 0; b < k; ++b)
    maskLow |= (1ull << layout.phys_of_log[logicalQubits[b]]);

  // Move complex pairs (re,im) together according to maskLow
  cast_permute_lowbits<ScalarType>(dCurrent, dScratch, nSys, maskLow, k, stream);
  std::swap(dCurrent, dScratch);

  set_layout_LSB(layout, logicalQubits, nSys);
}
#endif // CAST_USE_CUDA

/* --- small helper to decide if there’s enough memory for a scratch buffer --- */
static bool canUsePermuteLSB(int nQubits, int precBits, size_t* pSVBytes=nullptr, size_t* pScratchBytes=nullptr) {
#ifdef CAST_USE_CUDA
  const size_t scalar = (precBits == 32) ? sizeof(float) : sizeof(double);
  const size_t nComplex = size_t(1) << nQubits;              // 2^n
  const size_t svBytes = size_t(2) * nComplex * scalar;      // interleaved (re,im)
  const size_t scratch  = svBytes;                           // same size for ping-pong
  if (pSVBytes) *pSVBytes = svBytes;
  if (pScratchBytes) *pScratchBytes= scratch;

  size_t freeB=0, totalB=0;
  if (cudaMemGetInfo(&freeB, &totalB) != cudaSuccess) return true;
  const size_t safety = 512ull << 20; // 512 MiB
  return (svBytes + scratch + safety) <= freeB;
#else
  (void)nQubits; (void)precBits; (void)pSVBytes; (void)pScratchBytes;
  return false;
#endif
}



int main(int argc, char** argv) {
  using namespace timeit;

  Args args = parseCLI(argc, argv);
  std::cout << "[Info] QASM: " << args.qasmFile
            << ", prec=" << args.precBits
            << ", block=" << args.blockSize
            << ", reps=" << args.reps
            << ", threads=" << args.nThreads
            << ", fuse=" << (args.fuse ? "on" : "off")
            << (args.perfCSV.empty() ? "" : (", perf-csv="+args.perfCSV))
            << "\n";

  // 1) Parse QASM → Circuit (timed; ephemeral)
  Timer parseTimer(args.reps);
  auto parseTR = parseTimer.timeit([&]() {
    auto tmp = cast::parseCircuitFromQASMFile(args.qasmFile);
    if (!tmp) std::exit(2);
  });
  printTimingStats("1) Parse+IR Time", parseTR);

  // Parse QASM again and keep the Circuit
  auto circuitOrErr = cast::parseCircuitFromQASMFile(args.qasmFile);
  if (!circuitOrErr) {
    std::cerr << "[Error] parseCircuitFromQASMFile failed\n";
    return 2;
  }
  auto circuit = circuitOrErr.takeValue();

  auto circuitGraphs = circuit.getAllCircuitGraphs();
  if (circuitGraphs.empty()) {
    std::cerr << "[Error] no circuit graphs produced by parser\n";
    return 2;
  }
  ir::CircuitGraphNode* graph = circuitGraphs[0];

  // CUDA-guided fusion
  if (args.fuse) {
    const std::size_t gatesBefore = graph->getAllGatesShared().size();
    const Precision prec = (args.precBits == 32) ? Precision::F32 : Precision::F64;

    bool fused = false;

    if (!args.perfCSV.empty()) {
      try {
        // Load GPU performance data → build CUDA cost model
        auto cache = std::make_unique<CUDAPerformanceCache>(args.perfCSV);
        auto model = std::make_unique<CUDACostModel>(std::move(cache), /*zeroTol*/1e-8);
        model->setQueryBlockSize(args.blockSize);
        model->setQueryPrecision(prec);

        auto cfg = std::make_unique<CUDAFusionConfig>(std::move(model),
                                                      args.blockSize, prec);
        cfg->sizeMin = 2;
        cfg->sizeMax = 5;
        cfg->enableMultiTraverse = true;
        cfg->swapTol = 1e-8;
        cfg->zeroTol = 1e-8;

        CUDAOptimizer opt;
        opt.setCUDAFusionConfig(std::move(cfg))
           .enableCanonicalization()
           .enableFusion()
           .enableCFO();

        // Run optimizer on the whole circuit (quiet logger by default)
        opt.run(circuit);
        fused = true;
        std::cout << "[Fusion] CUDA adaptive fusion using CSV\n";
      } catch (const std::exception& e) {
        std::cerr << "[Fusion] ERROR: cannot load '" << args.perfCSV
                  << "': " << e.what() << "\n"
                  << "         Skipping fusion.\n";
      }
    } else {
      std::cout << "[Fusion] No --perf-csv provided; skipping CUDA fusion.\n";
    }

    // Refresh the graph pointer after IR mutation
    if (fused) {
      circuitGraphs = circuit.getAllCircuitGraphs();
      if (circuitGraphs.empty()) {
        std::cerr << "[Error] fusion produced no circuit graphs\n";
        return 2;
      }
      graph = circuitGraphs[0];

      const std::size_t gatesAfter = graph->getAllGatesShared().size();
      std::cout << "[Fusion] gates: " << gatesBefore << " -> " << gatesAfter << "\n";
    }
  }

  // Decide now if we can afford permute (needs full-size scratch).
  size_t svBytes=0, scratchBytes=0;
  bool usePermute = canUsePermuteLSB(graph->nQubits(), args.precBits, &svBytes, &scratchBytes);
  std::cout << "[Permute] SV=" << (svBytes >> 20) << " MiB, Scratch="
            << (scratchBytes >> 20) << " MiB, mode="
            << (usePermute ? "permute-to-LSB" : "no-permute (direct addressing)")
            << "\n";

  // 2) Kernel generation (timed)
  CUDAKernelGenConfig genCfg;
  genCfg.blockSize      = args.blockSize;
  genCfg.precision      = (args.precBits==32 ? cast::Precision::F32 : cast::Precision::F64);
  genCfg.matrixLoadMode = cast::CUDAMatrixLoadMode::UseMatImmValues;
  genCfg.assumeContiguousTargets = usePermute;  // LSB target assumption iff we can permute

  Timer kgTimer(args.reps);
  auto kgTR = kgTimer.timeit([&]() {
    cast::CUDAKernelManager km;
    auto r = km.genGraphGates(genCfg, *graph, args.graphName);
    if (!r) std::exit(3);
  });
  printTimingStats("2) Kernel Gen Time", kgTR);

  // 3) PTX emission (timed)
  Timer ptxTimer(args.reps);
  std::unique_ptr<cast::CUDAKernelManager> kmPTX;
  auto ptxTR = ptxTimer.timeitPartial(
    // pre:
    [&](){
      kmPTX = std::make_unique<cast::CUDAKernelManager>();
      auto r = kmPTX->genGraphGates(genCfg, *graph, args.graphName);
      if (!r) std::exit(4);
    },
    // timed:
    [&](){ kmPTX->emitPTX(args.nThreads, args.opt, 0); },
    // post:
    [&](){ kmPTX.reset(); },
    [](){}, [](){}
  );
  printTimingStats("3) PTX Time", ptxTR);

  // 4) JIT init (timed)
  Timer jitTimer(args.reps);
  std::unique_ptr<cast::CUDAKernelManager> kmJIT;
  auto jitTR = jitTimer.timeitPartial(
    // pre:
    [&](){
      kmJIT = std::make_unique<cast::CUDAKernelManager>();
      auto r = kmJIT->genGraphGates(genCfg, *graph, args.graphName);
      if (!r) std::exit(5);
      kmJIT->emitPTX(args.nThreads, args.opt, 0);
    },
    // timed:
    [&](){ kmJIT->initCUJIT(args.nThreads, 0); },
    // post:
    [&](){ kmJIT.reset(); },
    [](){}, [](){}
  );
  printTimingStats("4) JIT Time", jitTR);

  // 5) Final manager used for execution
  cast::CUDAKernelManager km;
  {
    auto r = km.genGraphGates(genCfg, *graph, args.graphName);
    if (!r) { std::cerr << "[Error] genGraphGates failed\n"; return 6; }
    km.emitPTX(args.nThreads, args.opt, 0);
    km.initCUJIT(args.nThreads, 0);
  }

  // 6) Collect kernels by reconstructing their names and execute
  std::vector<const CUDAKernelInfo*> kernels;
  {
    const std::string prefix = cast::internal::mangleGraphName(args.graphName);

    auto gates = graph->getAllGatesShared(); // vector<shared_ptr<...>>
    kernels.reserve(gates.size());
    size_t order = 0;
    for (const auto& g : gates) {
      // const std::string kname = prefix + "_" +
      //                           std::to_string(order++) + "_" +
      //                           std::to_string(graph->gateId(g));
      // if (const auto* ki = km.getKernelByName(kname)) {
      //   kernels.push_back(ki);
      // } else {
      //   std::cerr << "[WARN] kernel not found: " << kname << "\n";
      // }
      const std::string base = prefix + "_" +
                               std::to_string(order++) + "_" +
                               std::to_string(graph->gateId(g));
      const std::string preferred = usePermute ? (base + "_lsb") : (base + "_gen");
      const std::string fallback = usePermute ? (base + "_gen") : (base + "_lsb");
      if (const auto* ki = km.getKernelByName(preferred)) {
        kernels.push_back(ki);
      } else if (const auto* kj = km.getKernelByName(fallback)) {
        std::cerr << "[WARN] preferred variant missing for " << base
                  << "; using " << fallback << "\n";
        kernels.push_back(kj);
      } else {
        std::cerr << "[WARN] kernel not found (neither variant): " << base << "\n";
      }
    }
    std::cout << "[LOG] Number of generated GPU kernels: " << kernels.size() << "\n";
  }

  // Execution (timed) — with optional **permutation to LSBs** before each kernel
  Timer execTimer(args.reps);
  auto execTR = execTimer.timeit([&]() {
#ifdef CAST_USE_CUDA
    if (args.precBits == 32) {
      cast::CUDAStatevector<float> sv(graph->nQubits());
      sv.initialize();

      if (usePermute) {
        // prepare layout + scratch
        BitLayout layout; layout.init(sv.nQubits());
        float* dCurrent = reinterpret_cast<float*>(sv.dData());
        float* dScratch = nullptr;
        {
          const uint64_t nComplex = 1ull << sv.nQubits();
          const size_t nScalars = size_t(2) * nComplex; // interleaved (re,im)
          cudaError_t st = cudaMalloc(&dScratch, nScalars * sizeof(float));
          if (st != cudaSuccess || !dScratch) {
            // std::cerr << "[Permute] cudaMalloc scratch failed ("
            //           << (st==cudaSuccess?"nullptr":"cudaError")
            //           << "). Falling back to non-permute path.\n";
            // // Fallback at runtime: rebuild kernels without LSB assumption.
            // CUDAKernelGenConfig alt = genCfg;
            // alt.assumeContiguousTargets = false;
            // cast::CUDAKernelManager km2;
            // auto r = km2.genGraphGates(alt, *graph, args.graphName);
            // if (!r) std::exit(8);
            // km2.emitPTX(args.nThreads, args.opt, 0);
            // km2.initCUJIT(args.nThreads, 0);
            // for (auto* k : kernels) {
            //   const auto* k2 = km2.getKernelByName(k->llvmFuncName);
            //   if (!k2) { std::cerr<<"[ERR] missing fallback kernel "<<k->llvmFuncName<<"\n"; std::exit(9); }
            //   km2.launchCUDAKernel(sv.dData(), sv.nQubits(), *k2, args.blockSize);
            // }
            // cudaDeviceSynchronize();
            // return;
            std::cerr << "[Permute] cudaMalloc scratch failed (likely OOM). "
                        "Falling back to existing _gen kernels.\n";
            std::vector<const CUDAKernelInfo*> genKernels;
            genKernels.reserve(kernels.size());
            for (auto* k : kernels) {
              std::string name = k->llvmFuncName;
              if (name.size() >= 4 && name.compare(name.size()-4, 4, "_lsb") == 0) {
                name.replace(name.size()-4, 4, "_gen");
              }
              const auto* k2 = km.getKernelByName(name);
              if (!k2) {
                std::cerr << "[ERR] missing _gen kernel for " << k->llvmFuncName << "\n";
                std::exit(9);
              }
              genKernels.push_back(k2);
            }
            for (auto* k2 : genKernels) {
              km.launchCUDAKernel(sv.dData(), sv.nQubits(), *k2, args.blockSize);
            }
            cudaDeviceSynchronize();
            return;
          }
        }
        for (auto* k : kernels) {
          const auto& Q = k->gate->qubits();
          permute_to_LSBs_if_needed<float>(dCurrent, dScratch, sv.nQubits(),
                                           Q, layout, /*stream*/0);
          km.launchCUDAKernel(static_cast<void*>(dCurrent),
                              sv.nQubits(), *k, args.blockSize);
        }
        cudaDeviceSynchronize();
        cudaFree(dScratch);
      } else {
        for (auto* k : kernels)
          km.launchCUDAKernel(sv.dData(), sv.nQubits(), *k, args.blockSize);
        cudaDeviceSynchronize();
      }
    } else {
      cast::CUDAStatevector<double> sv(graph->nQubits());
      sv.initialize();

      if (usePermute) {
        BitLayout layout; layout.init(sv.nQubits());
        double* dCurrent = reinterpret_cast<double*>(sv.dData());
        double* dScratch = nullptr;
        {
          const uint64_t nComplex = 1ull << sv.nQubits();
          const size_t nScalars = size_t(2) * nComplex;
          cudaError_t st = cudaMalloc(&dScratch, nScalars * sizeof(double));
          if (st != cudaSuccess || !dScratch) {
            // std::cerr << "[Permute] cudaMalloc scratch failed (likely OOM). "
            //              "Falling back to non-permute path.\n";
            // CUDAKernelGenConfig alt = genCfg;
            // alt.assumeContiguousTargets = false;
            // cast::CUDAKernelManager km2;
            // auto r = km2.genGraphGates(alt, *graph, args.graphName);
            // if (!r) std::exit(8);
            // km2.emitPTX(args.nThreads, args.opt, 0);
            // km2.initCUJIT(args.nThreads, 0);
            // for (auto* k : kernels) {
            //   const auto* k2 = km2.getKernelByName(k->llvmFuncName);
            //   if (!k2) { std::cerr<<"[ERR] missing fallback kernel "<<k->llvmFuncName<<"\n"; std::exit(9); }
            //   km2.launchCUDAKernel(sv.dData(), sv.nQubits(), *k2, args.blockSize);
            // }
            // cudaDeviceSynchronize();
            // return;
            std::cerr << "[Permute] cudaMalloc scratch failed (likely OOM). "
                        "Falling back to existing _gen kernels.\n";
            std::vector<const CUDAKernelInfo*> genKernels;
            genKernels.reserve(kernels.size());
            for (auto* k : kernels) {
              std::string name = k->llvmFuncName;
              if (name.size() >= 4 && name.compare(name.size()-4, 4, "_lsb") == 0) {
                name.replace(name.size()-4, 4, "_gen");
              }
              const auto* k2 = km.getKernelByName(name);
              if (!k2) {
                std::cerr << "[ERR] missing _gen kernel for " << k->llvmFuncName << "\n";
                std::exit(9);
              }
              genKernels.push_back(k2);
            }
            for (auto* k2 : genKernels) {
              km.launchCUDAKernel(sv.dData(), sv.nQubits(), *k2, args.blockSize);
            }
            cudaDeviceSynchronize();
            return;
          }
        }
        if (!kernels.empty()) {
          std::cout << "[LOG] First kernel picked: " << kernels.front()->llvmFuncName << "\n";
        }
        for (auto* k : kernels) {
          const auto& Q = k->gate->qubits();
          permute_to_LSBs_if_needed<double>(dCurrent, dScratch, sv.nQubits(),
                                            Q, layout, /*stream*/0);
          km.launchCUDAKernel(static_cast<void*>(dCurrent),
                              sv.nQubits(), *k, args.blockSize);
        }
        cudaDeviceSynchronize();
        cudaFree(dScratch);
      } else {
        for (auto* k : kernels)
          km.launchCUDAKernel(sv.dData(), sv.nQubits(), *k, args.blockSize);
        cudaDeviceSynchronize();
      }
    }
#else
    (void)km; (void)kernels; // silence unused warnings when CUDA is off
    std::cerr << "[Error] Built without CUDA support.\n"; std::exit(7);
#endif
  });
  printTimingStats("5) Execution Time", execTR);

  // Summary
  std::cout << "======== Timing Results ========\n";
  printTimingStats("1) Parse+IR Time", parseTR);
  printTimingStats("2) Kernel Gen Time", kgTR);
  printTimingStats("3) PTX Time", ptxTR);
  printTimingStats("4) JIT Time", jitTR);
  printTimingStats("5) Execution Time", execTR);
  std::cout << "================================\n";

  return 0;
}
