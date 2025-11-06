#include "cast/CUDA/CUDA.h"
#include "cast/Core/Precision.h"
#include "utils/CSVParsable.h"

#include "llvm/Support/CommandLine.h"

#include <filesystem>

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
  float gen_time;
  float exec_wall_time;
  float exec_gpu_time;

  // clang-format off
  CSV_DATA_FIELD(device_name, num_threads, benchmark_name, num_qubits,
                 fusion_opt_level, precision,
                 num_u3, num_cx, num_gates_after_fusion, 
                 parse_time, opt_time, gen_time, exec_wall_time, exec_gpu_time)
  // clang-format on
};

using namespace cast;
namespace cl = llvm::cl;

static cl::OptionCategory Category("CUDA Profile Experiment Options");

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
  cl::desc("Path to the cuda cost model file."),
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
  using duration = std::chrono::duration<float>;
  std::vector<ProfileStats> stats;

  std::string deviceName = ArgDeviceName;
  int nThreads = cast::get_cpu_num_threads();
  cast::CUDAKernelManager km;
  CUDAStatevectorFP64 sv(getNumQubitsSV());
  sv.initialize();

  int count = 0;
  const auto run = [&](const std::string& inputFilename,
                       FusionOptLevel fusionOpt,
                       Precision precision) {
    // t0 to t1: parse circuit and optimizer setup
    auto t0 = clock::now();
    auto t1 = clock::now();

    // [time] parsing + optimizer setup
    t0 = clock::now();
    auto circuit = llvm::cantFail(parseCircuitFromQASMFile(inputFilename));
    CUDAOptimizer opt;
    opt.enableCFO(false);
    if (auto e = opt.loadCUDACostModelFromFile(ArgCostModel, precision)) {
      logerr() << "Failed to load CUDA cost model from file '" << ArgCostModel
               << "': " << llvm::toString(std::move(e)) << "\n";
      std::exit(1);
    }
    opt.getFusionConfig()->setOptLevel(fusionOpt);
    auto* cg = circuit->getAllCircuitGraphs()[0];
    t1 = clock::now();
    auto parse_time = duration(t1 - t0).count();

    // [log only] number of gates
    int nU3 = 0;
    int nCX = 0;
    auto allGates = cg->getAllGates();
    for (const auto* gate : allGates) {
      if (gate->nQubits() == 1)
        nU3++;
      else if (gate->nQubits() == 2)
        nCX++;
    }

    // [time] optimization
    t0 = clock::now();
    opt.run(*cg, {std::cerr, static_cast<int>(ArgVerbose)});
    t1 = clock::now();
    auto opt_time = duration(t1 - t0).count();

    // [log only] clean up previous kernels
    CUDAKernelGenConfig gConfig(precision);
    if (count > 0) {
      // clean up previous graph kernels and execution results
      km.clearGraphKernels("graph_" + std::to_string(count - 1));
      km.clearExecutionResults();
    }
    auto graphName = "graph_" + std::to_string(count++);

    // [time] kernel generation
    t0 = clock::now();
    if (auto e = km.genGraphGates(gConfig, *cg, graphName)) {
      logerr() << "Failed to generate CUDA kernels for graph '" << graphName
               << "': " << llvm::toString(std::move(e)) << "\n";
      std::exit(1);
    }
    t1 = clock::now();
    auto gen_time = duration(t1 - t0).count();

    // [time] JIT launch + execution

    t0 = clock::now();
    km.setLaunchConfig(sv.getDevicePtr(), cg->nQubits());
    auto ers = km.enqueueKernelLaunchesFromGraph(graphName);
    km.syncKernelExecution();
    t1 = clock::now();
    auto exec_wall_time = duration(t1 - t0).count();

    // [logging] collect gpu time
    float exec_gpu_time = 0.0f;
    for (const auto* er : ers)
      exec_gpu_time += er->getKernelTime();

    std::filesystem::path path(inputFilename);
    stats.push_back({.device_name = deviceName,
                     .num_threads = nThreads,
                     .benchmark_name = path.filename().string(),
                     .num_qubits = cg->nQubits(),
                     .fusion_opt_level = fusionOpt,
                     .precision = precision,
                     .num_u3 = nU3,
                     .num_cx = nCX,
                     .num_gates_after_fusion = static_cast<int>(cg->nGates()),
                     .parse_time = parse_time,
                     .opt_time = opt_time,
                     .gen_time = gen_time,
                     .exec_wall_time = exec_wall_time,
                     .exec_gpu_time = exec_gpu_time});
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