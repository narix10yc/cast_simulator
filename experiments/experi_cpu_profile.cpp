#include "cast/CPU/CPU.h"
#include "utils/CSVParsable.h"

#include "llvm/Support/CommandLine.h"

#include <filesystem>
#include <iostream>

struct ProfileStats : utils::CSVParsable<ProfileStats> {
  std::string device_name;
  int num_threads;
  std::string benchmark_name;
  int num_qubits;
  cast::FusionOptLevel fusion_opt_level;
  cast::Precision precision;

  int num_u3;
  int num_cx;
  int num_gates_after_fusion;

  float parse_time;
  float opt_time;
  float kernel_gen_time;
  float jit_compile_time;
  float exec_time;

  // clang-format off
  CSV_DATA_FIELD(device_name, num_threads, benchmark_name, num_qubits,
                 fusion_opt_level, precision,
                 num_u3, num_cx, num_gates_after_fusion,
                 parse_time, opt_time, kernel_gen_time, jit_compile_time,
                 exec_time)
  // clang-format on
};

using namespace cast;
namespace cl = llvm::cl;

static cl::OptionCategory Category("CPU Instrumental Profiling Options");

// clang-format off

static cl::opt<std::string>
ArgDeviceName("device", cl::cat(Category),
  cl::desc("Device name"), cl::Required);

static cl::opt<std::string>
ArgOutputFilename("o", cl::cat(Category),
  cl::desc("Output file name. If not provided, results are printed to stdout"),
  cl::init(""));

static cl::list<std::string>
ArgInputFilenames(cl::Positional, cl::cat(Category),
  cl::desc("List of input qasm files"),
  cl::CommaSeparated,
  cl::OneOrMore);

static cl::list<int>
ArgPrecisions("precision", cl::cat(Category),
  cl::desc("List of precisions to benchmark. Supported values: 32, 64"),
  cl::CommaSeparated,
  cl::OneOrMore);

static cl::list<std::string>
ArgOptModes("fusion", cl::cat(Category),
  cl::desc("A list of fusion optimization modes (mild, balanced, aggressive). "
           "Default to 'balanced'."),
  cl::CommaSeparated,
  cl::OneOrMore);

static cl::opt<std::string>
ArgCostModel("model", cl::cat(Category),
  cl::desc("Path to the CPU cost model file."),
  cl::Required);

static cl::opt<int>
ArgVerbose("verbose", cl::cat(Category),
  cl::desc("Verbosity level"), cl::init(1));

// clang-format on

static int getNumQubitsSV() {
  int nQubitsSV = 0;
  for (const auto& filename : ArgInputFilenames) {
    auto circuit = llvm::cantFail(parseCircuitFromQASMFile(filename));
    auto* cg = circuit->getAllCircuitGraphs()[0];
    nQubitsSV = std::max(nQubitsSV, cg->nQubits());
  }
  return nQubitsSV;
}

static std::ostream& logerr() { return std::cerr << RED("[Err] "); }

int main(int argc, char** argv) {
  cl::HideUnrelatedOptions(Category);
  cl::ParseCommandLineOptions(argc, argv);

  using clock = std::chrono::steady_clock;
  std::vector<ProfileStats> stats;

  std::string deviceName = ArgDeviceName;
  int nThreads = cast::get_cpu_num_threads();
  cast::CPUKernelManager km;
  CPUStatevectorF64 sv(getNumQubitsSV(), cast::get_cpu_simd_width());

  int count = 0;
  const auto run = [&](const std::string& inputFilename,
                       FusionOptLevel fusionOpt,
                       Precision precision) {
    // t0 to t1: parsing circuit and loading optimizer
    CPUKernelGenConfig gConfig(precision);
    auto graphName = "graph_" + std::to_string(count++);

    auto t0 = clock::now();
    auto circuit = llvm::cantFail(parseCircuitFromQASMFile(inputFilename));
    CPUOptimizer opt;
    opt.enableCFO(false);
    llvm::cantFail(opt.loadCPUCostModel(ArgCostModel, nThreads, precision));
    opt.getFusionConfig()->setOptLevel(fusionOpt);
    auto* cg = circuit->getAllCircuitGraphs()[0];
    auto t1 = clock::now();

    int nU3 = 0;
    int nCX = 0;
    const auto allGates = cg->getAllGates();
    for (const auto* gate : allGates) {
      if (gate->nQubits() == 1)
        nU3++;
      else
        nCX++;
    }

    // t2 to t3: circuit optimization (fusion, etc.)
    auto t2 = clock::now();
    opt.run(*cg, {std::cerr, ArgVerbose});
    auto t3 = clock::now();

    // t3 to t4: gate generation
    llvm::cantFail(km.genGraphGates(gConfig, *cg, graphName));
    auto t4 = clock::now();

    // t4 to t5: JIT compilation
    llvm::cantFail(km.compilePool(graphName));
    auto t5 = clock::now();

    // t5 to t6: execution
    llvm::cantFail(km.applyCPUKernelsFromGraph(
        sv.data(), cg->nQubits(), graphName, nThreads));
    auto t6 = clock::now();

    std::filesystem::path path(inputFilename);

    stats.push_back(
        {.device_name = deviceName,
         .num_threads = nThreads,
         .benchmark_name = path.filename().string(),
         .num_qubits = cg->nQubits(),
         .precision = precision,
         .num_u3 = nU3,
         .num_cx = nCX,
         .num_gates_after_fusion = static_cast<int>(cg->nGates()),
         .parse_time = std::chrono::duration<float>(t1 - t0).count(),
         .opt_time = std::chrono::duration<float>(t3 - t2).count(),
         .kernel_gen_time = std::chrono::duration<float>(t4 - t3).count(),
         .jit_compile_time = std::chrono::duration<float>(t5 - t4).count(),
         .exec_time = std::chrono::duration<float>(t6 - t5).count()});
    stats.back().write(std::cout);
    std::cout << "\n";
  };

  // Main loop starts here
  std::cout << ProfileStats::CSV_TITLE << "\n";
  for (const auto& optMode : ArgOptModes) {
    FusionOptLevel fusionOpt;
    if (optMode == "mild")
      fusionOpt = FusionOptLevel::Mild;
    else if (optMode == "balanced")
      fusionOpt = FusionOptLevel::Balanced;
    else if (optMode == "aggressive")
      fusionOpt = FusionOptLevel::Aggressive;
    else {
      logerr() << "Unknown fusion optimization mode: " << optMode << "\n";
      std::exit(1);
    }
    for (auto p : ArgPrecisions) {
      Precision prec;
      if (p == 32)
        prec = Precision::FP32;
      else if (p == 64)
        prec = Precision::FP64;
      else {
        logerr() << "Unsupported precision: " << p << "\n";
        std::exit(1);
      }

      for (const auto& inputFilename : ArgInputFilenames) {
        run(inputFilename, fusionOpt, prec);
      }
    }
  }

  if (!ArgOutputFilename.empty()) {
    std::ofstream ofs(ArgOutputFilename);
    if (!ofs) {
      logerr() << "Failed to open output file: " << ArgOutputFilename << "\n";
      return 1;
    }
    ofs << ProfileStats::CSV_TITLE << "\n";
    for (const auto& stat : stats) {
      stat.write(ofs);
      ofs << "\n";
    }
  }

  return 0;
}