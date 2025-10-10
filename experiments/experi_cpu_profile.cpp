#include "utils/CSVParsable.h"
#include "cast/Core/Precision.h"
#include "cast/CPU/CPU.h"

#include "llvm/Support/CommandLine.h"

struct ProfileStats : utils::CSVParsable<ProfileStats> {
  std::string device_name;
  int num_threads;
  std::string benchmark_name;
  int num_qubits;
  cast::Precision precision;
  float parse_opt_time;
  float jit_launch_time;
  float sv_init_time;
  float exec_time;

  // clang-format off
  CSV_DATA_FIELD(device_name, num_threads, benchmark_name, num_qubits,
                 precision, parse_opt_time, jit_launch_time, sv_init_time,
                 exec_time)
  // clang-format on
};

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
ArgOptMode("fusion", cl::cat(Category),
  cl::desc("A list of fusion optimization modes (mild, balanced, aggressive). "
           "Default to 'balanced'."),
  cl::init("balanced"));

static cl::opt<int>
ArgVerbose("verbose", cl::cat(Category),
  cl::desc("Verbosity level"), cl::init(1));

// clang-format on

int main(int argc, char** argv) {
  cl::HideUnrelatedOptions(Category);
  cl::ParseCommandLineOptions(argc, argv);

  using clock = std::chrono::steady_clock;
  std::vector<ProfileStats> stats;

  std::string deviceName = ArgDeviceName;
  int nThreads = cast::get_cpu_num_threads();
  cast::CPUKernelManager km;

  const auto run = [&]() {

  }


  return 0;
}