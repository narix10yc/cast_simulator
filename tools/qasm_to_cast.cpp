#include "cast/Fusion.h"
#include "cast/Legacy/CircuitGraph.h"
#include "cast/Legacy/ir_generator.h"
#include "cast/cpu.h"
#include "openqasm/parser.h"

#include "utils/iocolor.h"
#include "utils/utils.h"
#include "llvm/Support/CommandLine.h"

using IRGeneratorConfig = simulation::IRGeneratorConfig;
// using AmpFormat = IRGeneratorConfig::AmpFormat;
using utils::timedExecute;

using namespace IOColor;
using namespace llvm;
using namespace cast;
using namespace simulation;

#ifdef __BMI2__
#define PDEP_DEFAULT true
#else
#define PDEP_DEFAULT false
#endif

static cl::opt<std::string>
    InputFileName(cl::desc("input file name"), cl::Positional, cl::Required);

static cl::opt<std::string>
    OutputDirectory("o", cl::desc("output directory"), cl::init(""));

static cl::opt<std::string>
    Precision("p", cl::desc("precision (f64 or f32)"), cl::init("f64"));

static cl::opt<int> Verbose("verbose", cl::desc("verbose level"), cl::init(1));

static cl::opt<bool>
    UseF32("f32", cl::desc("use f32 (override -p)"), cl::init(false));

static cl::opt<int>
    simd_s("S", cl::desc("vector size (s value)"), cl::Prefix, cl::init(1));

static cl::opt<bool> MultiThreaded("multi-thread",
                                   cl::desc("enable multi-threading"),
                                   cl::init(true));

static cl::opt<bool>
    InstallTimer("timer", cl::desc("install timer"), cl::init(false));

static cl::opt<int>
    DebugLevel("debug", cl::desc("IR generation debug level"), cl::init(0));

// Gate Fusion Category
cl::OptionCategory GateCPUFusionConfigCategory("Gate Fusion Options", "");

static cl::opt<int> FusionLevel(
    "fusion",
    cl::cat(GateCPUFusionConfigCategory),
    cl::desc("fusion level presets 0 (disable), 1 (two-qubit only), "
             "2 (default), and 3 (aggressive)"),
    cl::init(2));

static cl::opt<int> MaxnQubits("max-k",
                               cl::cat(GateCPUFusionConfigCategory),
                               cl::desc("maximum number of qubits of gates"),
                               cl::init(0));

static cl::opt<int> MaxOpCount("max-op",
                               cl::cat(GateCPUFusionConfigCategory),
                               cl::desc("maximum operation count"),
                               cl::init(0));

static cl::opt<double> ZeroSkipThreshold("zero-thres",
                                         cl::cat(GateCPUFusionConfigCategory),
                                         cl::desc("zero skipping threshold"),
                                         cl::init(1e-8));

static cl::opt<bool> AllowMultipleTraverse(
    "allow-multi-traverse",
    cl::cat(GateCPUFusionConfigCategory),
    cl::desc("allow multiple tile traverse in gate fusion"),
    cl::init(true));

static cl::opt<bool>
    EnableIncrementScheme("increment-scheme",
                          cl::cat(GateCPUFusionConfigCategory),
                          cl::desc("enable increment fusion scheme"),
                          cl::init(true));

// IR Generation Category
cl::OptionCategory IRGenerationConfigCategory("IR Generation Options", "");

static cl::opt<bool> LoadMatrixInEntry("load-matrix-in-entry",
                                       cl::cat(IRGenerationConfigCategory),
                                       cl::desc("load matrix in entry"),
                                       cl::init(true));

static cl::opt<bool> LoadVectorMatrix("load-vector-matrix",
                                      cl::cat(IRGenerationConfigCategory),
                                      cl::desc("load vector matrix"),
                                      cl::init(false));

static cl::opt<bool> UseFMA("use-fma",
                            cl::cat(IRGenerationConfigCategory),
                            cl::desc("use fma (fused multiplication addition)"),
                            cl::init(true));

static cl::opt<bool>
    UseFMS("use-fms",
           cl::cat(IRGenerationConfigCategory),
           cl::desc("use fms (fused multiplication subtraction)"),
           cl::init(true));

static cl::opt<bool> UsePDEP("use-pdep",
                             cl::cat(IRGenerationConfigCategory),
                             cl::desc("use pdep (parallel bit deposit)"),
                             cl::init(PDEP_DEFAULT));

static cl::opt<bool>
    EnablePrefetch("enable-prefetch",
                   cl::cat(IRGenerationConfigCategory),
                   cl::desc("enable prefetch (not tested, recommend off)"),
                   cl::init(false));

static cl::opt<std::string>
    AmpFormat("amp-format",
              cl::cat(IRGenerationConfigCategory),
              cl::desc("amplitude format (reccommand 'alt')"),
              cl::init("alt"));

static cl::opt<double> ShareMatrixElemThres(
    "share-matrix-elem-thres",
    cl::cat(IRGenerationConfigCategory),
    cl::desc("share matrix element threshold (set to 0.0 to turn off)"),
    cl::init(0.0));

static cl::opt<bool> ShareMatrixElemUseImmValue(
    "share-matrix-elem-use-imm",
    cl::cat(IRGenerationConfigCategory),
    cl::desc("use immediate value for shared matrix elements"),
    cl::init(false));

static cl::opt<bool> ForceDenseKernel("force-dense-kernel",
                                      cl::cat(IRGenerationConfigCategory),
                                      cl::desc("force all kernels to be dense"),
                                      cl::init(false));

static cl::opt<bool>
    DumpIRToMultipleFiles("dump-ir-to-multiple-files",
                          cl::cat(IRGenerationConfigCategory),
                          cl::desc("dump ir to multiple files"),
                          cl::init(true));

static cl::opt<bool>
    WriteRawIR("write-raw-ir",
               cl::cat(IRGenerationConfigCategory),
               cl::desc("write raw ir files instead of bitcodes"),
               cl::init(false));

static cl::opt<int>
    NThreadsKernelGeneration("nthread-kernel-gen",
                             cl::cat(IRGenerationConfigCategory),
                             cl::desc("multi-thread kernel generation"),
                             cl::init(1));

static cl::opt<bool> RemoveFilesInsideOutputDirectory(
    "rm-out-dir",
    cl::cat(IRGenerationConfigCategory),
    cl::desc("remove files inside output directory"),
    cl::init(true));

int main(int argc, const char** argv) {
  legacy::CircuitGraph graph;
  FusionConfig fusionConfig;
  // CodeGeneratorCPU codeGenerator;
  IRGeneratorConfig irConfig;
  CodeGeneratorCPUConfig cpuConfig;
  cl::ParseCommandLineOptions(argc, argv);

  irConfig = IRGeneratorConfig{
      .simd_s = simd_s,
      .precision = (UseF32 || Precision == "f32") ? 32 : 64,
      .ampFormat = ((AmpFormat == "sep") ? IRGeneratorConfig::SepFormat
                                         : IRGeneratorConfig::AltFormat),
      .useFMA = UseFMA,
      .useFMS = UseFMS,
      .usePDEP = UsePDEP,
      .loadMatrixInEntry = LoadMatrixInEntry,
      .loadVectorMatrix = LoadVectorMatrix,
      .forceDenseKernel = ForceDenseKernel,
      .zeroSkipThres = ZeroSkipThreshold,
      .shareMatrixElemThres = ShareMatrixElemThres,
      .shareMatrixElemUseImmValue = ShareMatrixElemUseImmValue};

  irConfig.display();

  // parse arguments
  // timedExecute([&]() {
  if (MaxnQubits > 0 || MaxOpCount > 0) {
    if (MaxnQubits == 0 || MaxOpCount == 0) {
      std::cerr << RED_FG << BOLD << "Argument Error: " << RESET
                << "need to provide both 'max-k' and 'max-op'\n";
      return 1;
    }
    fusionConfig.maxnQubits = MaxnQubits;
    fusionConfig.maxOpCount = MaxOpCount;
    fusionConfig.zeroSkippingThreshold = ZeroSkipThreshold;
  } else
    fusionConfig = FusionConfig::Preset(FusionLevel);
  fusionConfig.allowMultipleTraverse = AllowMultipleTraverse;
  fusionConfig.incrementScheme = EnableIncrementScheme;

  if (OutputDirectory != "") {
    cpuConfig = CodeGeneratorCPUConfig{
        .multiThreaded = MultiThreaded,
        .installTimer = InstallTimer,
        .writeRawIR = WriteRawIR,
        .dumpIRToMultipleFiles = DumpIRToMultipleFiles,
        .rmFilesInsideOutputDirectory = RemoveFilesInsideOutputDirectory,
        .forceInOrder = false,
        .irConfig = irConfig,
    };

    // codeGenerator = CodeGeneratorCPU(cpuConfig, OutputFileName);
    // if (Verbose > 0) {
    //     codeGenerator.displayConfig(Verbose, std::cerr);
    //     cpuConfig.irConfig.checkConfliction(std::cerr);
    // }
  }
  // }, "Arguments parsed");

  if (Verbose > 0) {
    std::cerr << "-- Input file:  " << InputFileName << "\n";
    std::cerr << "-- Output directory: " << OutputDirectory << "\n";
  }

  if (Verbose > 0)
    fusionConfig.display(std::cerr);

  // parse and write ast
  timedExecute(
      [&]() {
        openqasm::Parser parser(InputFileName, 0);
        graph = parser.parse()->toCircuitGraph();
      },
      "Qasm AST Parsed");

  if (Verbose > 2)
    graph.print(std::cerr << "CircuitGraph Before Fusion:\n");
  if (Verbose > 0)
    graph.displayInfo(std::cerr, 2);

  // gate fusion
  timedExecute([&]() { cast::applyGateFusion(fusionConfig, graph); },
               "Gate Fusion Complete");
  graph.relabelBlocks();

  if (Verbose > 2) {
    graph.print(std::cerr << "CircuitGraph After Fusion:\n");
  }
  if (Verbose > 0)
    graph.displayInfo(std::cerr, Verbose + 1);

  if (OutputDirectory != "") {
    timedExecute(
        [&]() {
          generateCpuIrForRecompilation(
              graph, OutputDirectory, cpuConfig, NThreadsKernelGeneration);
        },
        "Generation complete!");
  }

  return 0;
}