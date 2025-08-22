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
#include "cast/CPU/CPUOptimizer.h"
#include "cast/CPU/CPUFusionConfig.h"
#include "cast/CPU/CPUCostModel.h"

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

static inline std::string trim(std::string s) {
  auto is_space = [](unsigned char c){return std::isspace(c)!=0;};
  while (!s.empty() && is_space(s.front())) s.erase(s.begin());
  while (!s.empty() && is_space(s.back()))  s.pop_back();
  return s;
}

static Precision parsePrecisionToken(const std::string& tok,
                                     Precision deflt = Precision::F64) {
  std::string t = tok;
  std::transform(t.begin(), t.end(), t.begin(),
                 [](unsigned char c){ return std::tolower(c); });
  if (t=="f32" || t=="32" || t=="fp32" || t=="float") return Precision::F32;
  if (t=="f64" || t=="64" || t=="fp64" || t=="double") return Precision::F64;
  return deflt;
}


static std::unique_ptr<CPUPerformanceCache>
loadPerfCacheFromCSV(const std::string& path,
                     Precision defaultPrecForRows = Precision::F64,
                     int verbose = 1) {
  auto cache = std::make_unique<CPUPerformanceCache>();

  std::ifstream ifs(path);
  if (!ifs) {
    std::cerr << "[Fusion] WARNING: cannot open perf CSV: " << path << "\n";
    return cache;
  }

  std::size_t added = 0, skipped = 0;
  std::string line;

  while (std::getline(ifs, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') { ++skipped; continue; }

    // Support both CSV and space-separated
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream iss(line);

    long long nQ = 0;
    unsigned long long opCount = 0;
    std::string precTok;
    long long nThreads = 0;
    double memSpd = 0.0;

    // Try: precision as token
    if (!(iss >> nQ >> opCount >> precTok >> nThreads >> memSpd)) {
      // Try: precision as integer
      iss.clear();
      iss.str(line);
      int precInt = 0;
      if (!(iss >> nQ >> opCount >> precInt >> nThreads >> memSpd)) {
        ++skipped;
        continue;
      }
      precTok = std::to_string(precInt);
    }

    if (nQ <= 0 || nThreads <= 0 || memSpd <= 0.0) {
      ++skipped;
      continue;
    }

    Precision prec = parsePrecisionToken(precTok, defaultPrecForRows);

    cache->items.emplace_back(static_cast<int>(nQ),
                              static_cast<uint64_t>(opCount),
                              prec,
                              static_cast<int>(nThreads),
                              memSpd);
    ++added;
  }

  if (verbose > 0) {
    std::cerr << "[Fusion] Loaded " << added << " perf rows"
              << " (skipped " << skipped << ") from " << path << "\n";
  }
  return cache;
}

struct Args {
  // std::string qasmFile = "../examples/qft/qft-16-cp.qasm";
  // std::string qasmFile = "../examples/rqc/q12_189_128.qasm";
  // std::string qasmFile = "../examples/rqc/q20_592_427.qasm";
  std::string qasmFile  = "../examples/rqc/q28_442_300.qasm";
  // std::string qasmFile = "../examples/rqc/q30_521_379.qasm";
  // std::string qasmFile  = "../examples/rqc/q30_4299_3272.qasm";
  std::string graphName = "testCircuit";
  int reps = 3;
  int blockSize = 256;
  int nThreads = 1;
  int precBits = 64;
  bool fuse = true;
  std::string perfCSV;
  llvm::OptimizationLevel opt = llvm::OptimizationLevel::O1;
};

static Args parseCLI(int argc, char** argv) {
  Args a;
  for (int i=1;i<argc;++i) {
    std::string v(argv[i]);
    auto next = [&](int& i)->std::string{
      if (i+1>=argc){ std::cerr<<"Missing value after "<<v<<"\n"; std::exit(1);}
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

  // If a CSV is provided, use adaptive cost model;
  // otherwise: fallback to size-only fusion.
  if (args.fuse) {
    const std::size_t gatesBefore = graph->getAllGatesShared().size();
    const Precision prec = (args.precBits == 32) ? Precision::F32 : Precision::F64;

    CPUOptimizer opt; // IR-level fusion (backend-agnostic)

    if (!args.perfCSV.empty()) {
      // Adaptive fusion path
      constexpr double zeroTol = 1e-8;
      constexpr double swapTol = 1e-8;
      auto cache = loadPerfCacheFromCSV(args.perfCSV, prec, 1);

      if (!cache->items.empty()) {
        auto model = std::make_unique<CPUCostModel>(std::move(cache), zeroTol);

        auto fcfg = std::make_unique<CPUFusionConfig>(
            std::move(model),
            std::max(args.nThreads, 1),
            prec);

        fcfg->sizeMin = 2;
        fcfg->sizeMax = 5;
        fcfg->enableMultiTraverse = true;
        fcfg->zeroTol = zeroTol;
        fcfg->swapTol = swapTol;

        opt.setCPUFusionConfig(std::move(fcfg))
           .enableCanonicalization()
           .enableFusion()
           .enableCFO();

        std::cout << "[Fusion] Adaptive fusion using CSV cost model\n";
      } else {
        std::cerr << "[Fusion] WARNING: CSV had no usable rows; "
                     "falling back to size-only fusion.\n";
        opt.setSizeOnlyFusionConfig(3)
           .enableCanonicalization()
           .enableFusion()
           .enableCFO();
      }
    } else {
      opt.setSizeOnlyFusionConfig(3)
         .enableCanonicalization()
         .enableFusion()
         .enableCFO();
      std::cout << "[Fusion] Size-only fusion (no CSV)\n";
    }

    // Run optimizer on the whole circuit
    opt.run(circuit);

    // Refresh the graph pointer after IR was mutated
    circuitGraphs = circuit.getAllCircuitGraphs();
    if (circuitGraphs.empty()) {
      std::cerr << "[Error] fusion produced no circuit graphs\n";
      return 2;
    }
    graph = circuitGraphs[0];

    const std::size_t gatesAfter = graph->getAllGatesShared().size();
    std::cout << "[Fusion] gates: " << gatesBefore << " -> " << gatesAfter << "\n";
  }

  // 2) Kernel generation (timed)
  CUDAKernelGenConfig genCfg;
  genCfg.blockSize      = args.blockSize;
  genCfg.precision      = (args.precBits==32 ? cast::Precision::F32 : cast::Precision::F64);
  genCfg.matrixLoadMode = cast::CUDAMatrixLoadMode::UseMatImmValues;

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
      const std::string kname = prefix + "_" +
                                std::to_string(order++) + "_" +
                                std::to_string(graph->gateId(g));
      if (const auto* ki = km.getKernelByName(kname)) {
        kernels.push_back(ki);
      } else {
        std::cerr << "[WARN] kernel not found: " << kname << "\n";
      }
    }
    std::cout << "[LOG] Number of generated GPU kernels: " << kernels.size() << "\n";
  }

  // Execution (timed)
  Timer execTimer(args.reps);
  auto execTR = execTimer.timeit([&]() {
    if (args.precBits == 32) {
      cast::CUDAStatevector<float> sv(graph->nQubits());
      sv.initialize();
      for (auto* k : kernels)
        km.launchCUDAKernel(sv.dData(), sv.nQubits(), *k, args.blockSize);
    } else {
      cast::CUDAStatevector<double> sv(graph->nQubits());
      sv.initialize();
      for (auto* k : kernels)
        km.launchCUDAKernel(sv.dData(), sv.nQubits(), *k, args.blockSize);
    }
    cudaDeviceSynchronize();
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
