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
#include <stdexcept>

#include <cuda_runtime.h>

#include "timeit/timeit.h"
#include "llvm/Passes/OptimizationLevel.h"

#include "cast/Core/IRNode.h"
#include "cast/Core/KernelGenInternal.h"

#include "cast/CUDA/CUDAKernelManager.h"
#include "cast/CUDA/CUDAStatevector.h"
#include "cast/CUDA/Config.h"

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

static inline void rtrim(std::string& s) {
  while (!s.empty() && (s.back()=='\r' || s.back()=='\n' || std::isspace(unsigned(s.back()))))
    s.pop_back();
}
static inline void ltrim(std::string& s) {
  size_t i=0; while (i<s.size() && std::isspace(unsigned(s[i]))) ++i;
  if (i) s.erase(0,i);
}
static inline std::string trim(std::string s){ rtrim(s); ltrim(s); return s; }
static inline std::string lower(std::string s){ std::transform(s.begin(), s.end(), s.begin(),
  [](unsigned char c){ return std::tolower(c); }); return s; }

static std::vector<std::string> splitCSV(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  cur.reserve(line.size());
  bool in_quote = false;
  for (size_t i=0;i<line.size();++i) {
    char c = line[i];
    if (c=='"') {
      in_quote = !in_quote;
    } else if (c==',' && !in_quote) {
      out.push_back(trim(cur));
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  out.push_back(trim(cur));
  return out;
}

static Precision parsePrecisionToken(std::string t) {
  t = lower(trim(t));
  if (t=="f32" || t=="32" || t=="float" || t=="single") return Precision::F32;
  if (t=="f64" || t=="64" || t=="double")               return Precision::F64;
  // numeric enum fallback
  try {
    int v = std::stoi(t);
    if (v == int(Precision::F32)) return Precision::F32;
    if (v == int(Precision::F64)) return Precision::F64;
  } catch (...) {}
  return Precision::Unknown;
}

static CUDAVariant parseVariantToken(std::string t) {
  t = lower(trim(t));
  if (t=="global" || t=="0") return CUDAVariant::Global;
  if (t=="smemtranspose" || t=="smem" || t=="transpose" || t=="1")
    return CUDAVariant::SmemTranspose;
  try { if (std::stoi(t)==1) return CUDAVariant::SmemTranspose; } catch(...) {}
  return CUDAVariant::Global;
}

static CUDAComputePath parsePathToken(std::string t) {
  t = lower(trim(t));
  if (t=="scalaralu" || t=="scalar" || t=="alu" || t=="0")
    return CUDAComputePath::ScalarALU;
  if (t=="tensorcore" || t=="tensor" || t=="tc" || t=="1")
    return CUDAComputePath::TensorCore;
  try { if (std::stoi(t)==1) return CUDAComputePath::TensorCore; } catch(...) {}
  return CUDAComputePath::ScalarALU;
}

static AccessPattern parsePatternToken(std::string t) {
  t = lower(trim(t));
  if (t=="contiguous" || t=="contig" || t=="lsb" || t=="0")
    return AccessPattern::Contiguous;
  if (t=="semi" || t=="1")
    return AccessPattern::Semi;
  if (t=="strided" || t=="msb" || t=="2")
    return AccessPattern::Strided;
  try {
    int v = std::stoi(t);
    if (v==0) return AccessPattern::Contiguous;
    if (v==1) return AccessPattern::Semi;
    if (v==2) return AccessPattern::Strided;
  } catch(...) {}
  return AccessPattern::Contiguous;
}

static std::unique_ptr<cast::CUDAPerformanceCache>
loadGpuPerfCSV(const std::string& path) {
  using Cache = cast::CUDAPerformanceCache;
  using Item  = cast::CUDAPerformanceCache::Item;

  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("Cannot open performance CSV '" + path + "'");
  }

  std::vector<Item> items;
  std::string line;

  // read first line and check header
  if (!std::getline(ifs, line)) {
    throw std::runtime_error("Empty performance CSV: " + path);
  }
  rtrim(line);
  const std::string header = Item::CSV_TITLE;
  if (line != header) {
    // no header: parse this line as data
    auto toks = splitCSV(line);
    if (toks.size() >= 20) {
      Item it{};
      size_t i=0;
      it.nQubitsGate = std::stoi(toks[i++]);
      it.opCount     = std::stoi(toks[i++]);
      it.precision   = parsePrecisionToken(toks[i++]);
      it.variant     = parseVariantToken(toks[i++]);
      it.path        = parsePathToken(toks[i++]);
      it.pattern     = parsePatternToken(toks[i++]);

      int gx = std::stoi(toks[i++]), gy = std::stoi(toks[i++]), gz = std::stoi(toks[i++]);
      int bx = std::stoi(toks[i++]), by = std::stoi(toks[i++]), bz = std::stoi(toks[i++]);
      it.gridDim  = dim3(gx, gy, gz);
      it.blockDim = dim3(bx, by, bz);

      it.regsPerThread = std::stoi(toks[i++]);
      it.smemPerBlock  = std::stoi(toks[i++]);
      it.occupancy     = std::stod(toks[i++]);
      it.time_s        = std::stod(toks[i++]);
      it.bytes         = std::stod(toks[i++]);
      it.flops         = std::stod(toks[i++]);
      it.Bach_GBs      = std::stod(toks[i++]);
      it.Pach_GFLOPs   = std::stod(toks[i++]);

      items.push_back(std::move(it));
    }
  }

  // parse remaining lines
  while (std::getline(ifs, line)) {
    rtrim(line);
    if (line.empty()) continue;
    auto toks = splitCSV(line);
    if (toks.size() < 20) continue; // skip malformed
    Item it{};
    size_t i=0;
    it.nQubitsGate = std::stoi(toks[i++]);
    it.opCount     = std::stoi(toks[i++]);
    it.precision   = parsePrecisionToken(toks[i++]);
    it.variant     = parseVariantToken(toks[i++]);
    it.path        = parsePathToken(toks[i++]);
    it.pattern     = parsePatternToken(toks[i++]);

    int gx = std::stoi(toks[i++]), gy = std::stoi(toks[i++]), gz = std::stoi(toks[i++]);
    int bx = std::stoi(toks[i++]), by = std::stoi(toks[i++]), bz = std::stoi(toks[i++]);
    it.gridDim  = dim3(gx, gy, gz);
    it.blockDim = dim3(bx, by, bz);

    it.regsPerThread = std::stoi(toks[i++]);
    it.smemPerBlock  = std::stoi(toks[i++]);
    it.occupancy     = std::stod(toks[i++]);
    it.time_s        = std::stod(toks[i++]);
    it.bytes         = std::stod(toks[i++]);
    it.flops         = std::stod(toks[i++]);
    it.Bach_GBs      = std::stod(toks[i++]);
    it.Pach_GFLOPs   = std::stod(toks[i++]);

    items.push_back(std::move(it));
  }

  return std::make_unique<Cache>(std::move(items));
}

// ------------------ CLI ------------------

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
    else if (v=="--prec")      a.precBits = std::stoi(next(i));     // 32/64
    else if (v=="--no-fusion") a.fuse = false;
    else if (v=="--perf-csv")  a.perfCSV = next(i);                 // GPU CSV
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

  // 1.5) CUDA-guided fusion (optional).
  if (args.fuse) {
    const std::size_t gatesBefore = graph->getAllGatesShared().size();
    const Precision prec = (args.precBits == 32) ? Precision::F32 : Precision::F64;

    bool fused = false;

    if (!args.perfCSV.empty()) {
      try {
        // Load GPU performance data → build CUDA cost model
        auto cache = loadGpuPerfCSV(args.perfCSV);
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

  // 6) Collect kernels (prefer _lsb, else _gen)
  // std::vector<const CUDAKernelInfo*> kernels;
  // {
  //   const std::string prefix = cast::internal::mangleGraphName(args.graphName);
  //   auto gates = graph->getAllGatesShared();
  //   kernels.reserve(gates.size());

  //   size_t order = 0;
  //   for (const auto& g : gates) {
  //     const std::string base = prefix + "_" +
  //                             std::to_string(order++) + "_" +
  //                             std::to_string(graph->gateId(g));

  //     const CUDAKernelInfo* ki = km.getKernelByName(base + "_gen");
  //     if (!ki) ki = km.getKernelByName(base + "_lsb");

  //     if (ki) {
  //       kernels.push_back(ki);
  //     } else {
  //       std::cerr << "[WARN] kernel not found (tried _gen/_lsb): " << base << "\n";
  //     }
  //   }
  //   std::cout << "[LOG] Number of generated GPU kernels: " << kernels.size() << "\n";
  // }

  // 6) Collect kernels (prefer _lsb, else _gen) and print fused gate size per kernel
  std::vector<const CUDAKernelInfo*> kernels;
  {
    const std::string prefix = cast::internal::mangleGraphName(args.graphName);
    auto gates = graph->getAllGatesShared();
    kernels.reserve(gates.size());

    std::map<int, std::size_t> sizeHistogram; // optional: summary by k

    size_t order = 0;
    for (const auto& g : gates) {
      const int k = static_cast<int>(g->nQubits());        // fused gate size
      const uint64_t dim = 1ull << k;                      // matrix dimension
      const auto& qs = g->qubits();                          // qubit indices

      const std::string base = prefix + "_" +
                              std::to_string(order) + "_" +
                              std::to_string(graph->gateId(g));

      std::string chosenName;
      const CUDAKernelInfo* ki = km.getKernelByName(base + "_gen");
      if (ki) {
        chosenName = base + "_gen";
      } else {
        ki = km.getKernelByName(base + "_lsb");
        if (ki) chosenName = base + "_lsb";
      }

      if (ki) {
        kernels.push_back(ki);
        ++sizeHistogram[k];

        // Print per-kernel fused gate size (and some context)
        std::cout << "[Kernel] #" << order
                  << " name=" << chosenName
                  << " gateId=" << graph->gateId(g)
                  << " k=" << k << " (dim=" << dim << "x" << dim << ") qubits=[";
        for (size_t i = 0; i < qs.size(); ++i) {
          std::cout << qs[i] << (i + 1 < qs.size() ? "," : "");
        }
        std::cout << "]\n";
      } else {
        std::cerr << "[WARN] kernel not found (tried _gen/_lsb): " << base << "\n";
      }

      ++order;
    }

    std::cout << "[LOG] Number of generated GPU kernels: " << kernels.size() << "\n";

    // Optional: print a histogram of fused gate sizes
    if (!sizeHistogram.empty()) {
      std::cout << "[LOG] Fused gate size histogram:\n";
      for (const auto& [k, cnt] : sizeHistogram) {
        std::cout << "  k=" << k << " : " << cnt << "\n";
      }
    }
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
