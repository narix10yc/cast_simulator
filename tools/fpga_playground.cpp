#include "cast/FPGA/FPGAInst.h"
#include "cast/FPGA/FPGAFusion.h"

#include "openqasm/parser.h"
#include "utils/CommandLine.h"

#include <chrono>
#include <cmath>

namespace cl = utils::cl;
using namespace cast;

static auto&
ArgInputFileName = cl::registerArgument<std::string>("i")
  .desc("Input file name")
  .setArgumentPositional();

static auto&
ArgOutputFilename = cl::registerArgument<std::string>("o")
  .desc("Output file name (end with .csv)")
  .setOccurExactlyOnce();

static auto&
ArgForceFilename = cl::registerArgument<bool>("force-name")
  .desc("Force output filename (possibly not ending with '.csv')")
  .init(false);

static auto&
ArgOverwriteMode = cl::registerArgument<bool>("overwrite")
  .desc("Overwrite the output file with new results")
  .init(false);

static auto&
ArgRecursiveMode = cl::registerArgument<bool>("r")
  .desc("Recursive mode")
  .init(false);

struct InstStatistics {
  int nNonExtMemInst = 0, nExtMemInst = 0, nSqGateInst = 0, nUpGateInst = 0;

  std::ostream& print(std::ostream& os) {
    int nGateInst = nSqGateInst + nUpGateInst;
    int nMemInst = nNonExtMemInst + nExtMemInst;
    return os << BOLDCYAN("====== Instruction Statistics: =======\n"
          "Num Instructions: " << nGateInst + nMemInst << "\n")
      << "  - num gate instructions:   " << nGateInst << "\n"
      << "    - num SQ gate instructions: " << nSqGateInst << "\n"
      << "    - num UP gate instructions: " << nUpGateInst << "\n"
      << "  - num memory instructions: " << nMemInst << "\n"
      << "    - num EXT mem instructions:     " << nExtMemInst << "\n"
      << "    - num non-EXT mem instructions: " << nNonExtMemInst << "\n"
      << " ----------------------------------- \n";
  }
};

struct InstCycleStatistics {
  int nTwiceExtMem = 0, nExtMem = 0, nNonExtMem = 0;
  int nGeneralSQGate = 0, nRealOnlySQGate = 0, nUPGate = 0;
  int nOverlappingInst = 0;

  std::ostream& print(std::ostream& os) {
    return os << BOLDCYAN("====== Cycle Statistics: =======\n")
      << "  - nTwiceExtMemTime: " << nTwiceExtMem << "\n"
      << "  - nExtMemTime:      " << nExtMem << "\n"
      << "  - nNonExtMemTime:   " << nNonExtMem << "\n"
      << "  - nGeneralSQGate:   " << nGeneralSQGate << "\n"
      << "  - nRealOnlySQGate:  " << nRealOnlySQGate << "\n"
      << "  - nUPGate:          " << nUPGate << "\n"
      << " ----------------------------------- \n";
  }
};

std::ostream& writCSVLine(std::ostream& os,
    const InstStatistics& instStats, const InstCycleStatistics& cycleStats) {
  return os << instStats.nSqGateInst << "," << instStats.nUpGateInst << ","
    << instStats.nExtMemInst << "," << instStats.nNonExtMemInst << ","
    << cycleStats.nTwiceExtMem << "," << cycleStats.nExtMem << ","
    << cycleStats.nNonExtMem << "," << cycleStats.nGeneralSQGate << ","
    << cycleStats.nRealOnlySQGate << "," << cycleStats.nUPGate << ","
    << cycleStats.nOverlappingInst;
}

InstStatistics getInstStatistics(const std::vector<fpga::Instruction>& insts) {
  InstStatistics stats;
  for (const auto& inst : insts) {
    if (inst._gInst->getKind() == GOp_SQ)
      stats.nSqGateInst++;
    else if (inst._gInst->getKind() == GOp_UP)
    stats.nUpGateInst++;
    if (!inst._mInst->isNull()) {
      if (inst._mInst->getKind() == MOp_EXT)
        stats.nExtMemInst++;
      else
        stats.nNonExtMemInst++;
    }
  }
  return stats;
}

InstCycleStatistics getCycleStatistics(
    const std::vector<fpga::Instruction>& insts,
    const FPGACostConfig& costConfig,
    bool additionalExtMemOp) {
  InstCycleStatistics stats;
  for (const auto& inst : insts) {
    // inst.print(std::cerr);
    if (!inst._mInst->isNull() && !inst._gInst->isNull())
      stats.nOverlappingInst++;
    auto costKind = inst.getCostKind(costConfig);
    switch (costKind) {
    case Instruction::CK_TwiceExtMemTime:
      stats.nTwiceExtMem++;
      break;
    case Instruction::CK_ExtMemTime:
      stats.nExtMem++;
      break;
    case Instruction::CK_NonExtMemTime:
      stats.nNonExtMem++;
      break;
    case Instruction::CK_GeneralSQGate:
      stats.nGeneralSQGate++;
      break;
    case Instruction::CK_RealOnlySQGate:
      stats.nRealOnlySQGate++;
      break;
    case Instruction::CK_UPGate:
      stats.nUPGate++;
      break;
    default:
      assert(false && "Unreachable");
      break;
    }
  }

  if (additionalExtMemOp)
    stats.nExtMem++;
  return stats;
}

int costKindToNumNormalizedCycle(Instruction::CostKind kind) {
  switch (kind) {
  case Instruction::CK_TwiceExtMemTime:
    return 84;
  case Instruction::CK_ExtMemTime:
    return 42;
  case Instruction::CK_NonExtMemTime:
    return 1;

  case Instruction::CK_GeneralSQGate:
    return 2;
  case Instruction::CK_RealOnlySQGate:
    return 1;
  case Instruction::CK_UPGate:
    return 1;
  default:
    assert(false && "Unreachable");
    return 0;
  }
}

void runExperiment(
    std::function<void(legacy::CircuitGraph&)> f,
    std::ostream& os, const std::string& circuitName) {
  int nLocalQubits = 14;
  int gridSize = 4;
  int nOnChipQubits = nLocalQubits + 2 * gridSize;
  FPGAFusionConfig fusionConfig{
      .maxUnitaryPermutationSize = 8,
      .ignoreSingleQubitNonCompGates = true,
      .multiTraverse = true,
      .tolerances = FPGAGateCategoryTolerance::Default,
  };

  FPGACostConfig costConfig{.lowestQIdxForTwiceExtTime = 7};

  FPGAInstGenConfig instGen11ConfigUp8{
      .nLocalQubits = nLocalQubits,
      .gridSize = gridSize,
      .selectiveGenerationMode = true,
      .maxUpSize = 8,
      .tolerances = FPGAGateCategoryTolerance::Default,
  };

  FPGAInstGenConfig instGen11Config{
      .nLocalQubits = nLocalQubits,
      .gridSize = gridSize,
      .selectiveGenerationMode = true,
      .maxUpSize = 5,
      .tolerances = FPGAGateCategoryTolerance::Default,
  };

  FPGAInstGenConfig instGen10Config{
      .nLocalQubits = nLocalQubits,
      .gridSize = gridSize,
      .selectiveGenerationMode = false,
      .maxUpSize = 5,
      .tolerances = FPGAGateCategoryTolerance::Default,
  };

  FPGAInstGenConfig instGen01Config{
      .nLocalQubits = nLocalQubits,
      .gridSize = gridSize,
      .selectiveGenerationMode = true,
      .maxUpSize = 0,
      .tolerances = FPGAGateCategoryTolerance::Default,
  };

  FPGAInstGenConfig instGen00Config{
      .nLocalQubits = nLocalQubits,
      .gridSize = gridSize,
      .selectiveGenerationMode = false,
      .maxUpSize = 0,
      .tolerances = FPGAGateCategoryTolerance::Default,
  };

  FPGAInstGenConfig instGenBadConfig{
      .nLocalQubits = nLocalQubits,
      .gridSize = gridSize,
      .selectiveGenerationMode = false,
      .maxUpSize = 0,
      .tolerances = FPGAGateCategoryTolerance::Zero,
  };

  const auto run = [&](const FPGAInstGenConfig& instGenConfig) {
    legacy::CircuitGraph graph;
    f(graph);
    auto tBegin = std::chrono::high_resolution_clock::now();
    auto instructions = fpga::genInstruction(graph, instGenConfig);
    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tInSec = std::chrono::duration_cast<std::chrono::duration<double>>(
      tEnd - tBegin).count();
    
    auto instStats = getInstStatistics(instructions);
    auto cycleStats = getCycleStatistics(
      instructions, costConfig, graph.nQubits > nOnChipQubits);
    writCSVLine(os, instStats, cycleStats) << "," << std::scientific << tInSec << "\n";
  };

  legacy::CircuitGraph tmpGraph;
  f(tmpGraph);
  auto nQubits = tmpGraph.nQubits;
  auto allBlocks = tmpGraph.getAllBlocks();
  auto nGates = allBlocks.size();

  int opCountMax = 0;
  int opCount = 0;
  for (const auto& b : allBlocks) {
    opCountMax += 1ULL << (2 * b->nQubits() + 1);
    opCount += b->quantumGate->opCount(1e-8);
  }

  // fusion, instGen, gateValueTolerance
  os << circuitName << "-up8," << nQubits << "," << nGates
     << "," << opCount << "," << opCountMax << ",true,true,true,";
  run(instGen11ConfigUp8);
  os << circuitName << "," << nQubits << "," << nGates
     << "," << opCount << "," << opCountMax << ",true,true,true,";
  run(instGen11Config);
  os << circuitName << "," << nQubits << "," << nGates
     << "," << opCount << "," << opCountMax << ",true,false,true,";
  run(instGen10Config);
  os << circuitName << "," << nQubits << "," << nGates
     << "," << opCount << "," << opCountMax << ",false,true,true,";
  run(instGen01Config);
  os << circuitName << "," << nQubits << "," << nGates
     << "," << opCount << "," << opCountMax << ",false,false,true,";
  run(instGen00Config);
  os << circuitName << "," << nQubits << "," << nGates
     << "," << opCount << "," << opCountMax << ",false,false,false,";
  run(instGenBadConfig);
}

// terminate on error
void checkOutputFileName() {
  if (ArgForceFilename)
    return;
  const std::string& fileName = ArgOutputFilename;
  if (fileName.length() > 4 && fileName.ends_with(".csv"))
    return;
  std::cerr << BOLDYELLOW("Notice: ")
            << "Output filename does not end with '.csv'. "
               "If this filename is desired, please add '-force-name' "
               "commandline option\n";
  std::exit(1);
}

void processQFT(std::ofstream& oFile, int nQubits) {
  std::string circuit = "QFT-" + std::to_string(nQubits);
  std::cerr << "Processing " << circuit << "\n";
  runExperiment([nQubits](legacy::CircuitGraph& graph) {
    legacy::CircuitGraph::QFTCircuit(nQubits, graph);
  }, oFile, circuit);
}

void processFile(const std::filesystem::path& path, std::ofstream& oFile) {
  std::cerr << "Processing " << path << "\n";
  if (path.extension() != ".qasm") {
    std::cerr << BOLDYELLOW("[Warning]: ")
              << "Ignored " << path
              << " because its name does not end with '.qasm'.\n";
    return;
  }

  runExperiment([path](legacy::CircuitGraph& graph) {
    openqasm::Parser qasmParser(path.string(), -1);
    qasmParser.parse()->toCircuitGraph(graph);
    // CircuitGraph::QFTCircuit(32, graph);
  }, oFile, path.filename().stem().string());
}

void openCSVToWriteOn(std::ofstream& oFile) {
  checkOutputFileName();
  if (ArgOverwriteMode)
    oFile.open(ArgOutputFilename, std::ios::out | std::ios::trunc);
  else
    oFile.open(ArgOutputFilename, std::ios::app);

  std::ifstream iFile(ArgOutputFilename);

  if (!oFile || !iFile) {
    std::cerr << BOLDRED("[Error]: ")
              << "Unable to open file '" << ArgOutputFilename << "'.\n";
    std::exit(1);
  }
  if (iFile.peek() == std::ifstream::traits_type::eof()) {
    oFile << "circuit,nQubits,nGates,opCount,opCountMax,"
      << "fusionOpt,instGenOpt,gateValueToleranceOpt,"
      << "nSqGateInst,nUpGateInst,nExtMemInst,nNonExtMemInst,"
      << "nTwiceExtMemTime,nExtMemTime,nNonExtMemTime,"
      << "nGeneralSQGateTime,nRealOnlySQGateTime,nUPGateTime,nOverlappingInstTime,"
      << "instGenTime\n";
  }
  iFile.close();
}

int main(int argc, char** argv) {
  cl::ParseCommandLineArguments(argc, argv);

  cl::DisplayArguments();

  std::ofstream oFile;
  openCSVToWriteOn(oFile);

  if (ArgRecursiveMode) {
    std::filesystem::path path(ArgInputFileName);
    if (!std::filesystem::is_directory(path)) {
      std::cerr << BOLDRED("[Error]: ")
                << "The input filename must be a directory"
                   " when recursive mode is turned on.\n";
      std::exit(1);
    }
    for (const auto& entry : std::filesystem::directory_iterator(path))
      processFile(entry.path(), oFile);
  } else {
    processFile(static_cast<std::string>(ArgInputFileName), oFile);
  }

  for (int q = 8; q < 41; q++) {
    processQFT(oFile, q);
  }

  return 0;
}
